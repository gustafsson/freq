#include "mergertexture.h"

#include "heightmap/blockquery.h"
#include "mergekernel.h"
#include "heightmap/render/shaderresource.h"
#include "heightmap/render/blocktextures.h"

#include "tasktimer.h"
#include "computationkernel.h"
#include "gltextureread.h"
#include "glframebuffer.h"
#include "datastoragestring.h"
#include "GlException.h"
#include "glPushContext.h"
#include "glprojection.h"
#include "gluperspective.h"
#include "glgroupmarker.h"
#include "glstate.h"

#include <QGLContext>

//#ifdef GL_ES_VERSION_2_0
#define DRAW_STRAIGHT_ONTO_BLOCK
//#endif

//#define VERBOSE_COLLECTION
#define VERBOSE_COLLECTION if(0)

//#define INFO_COLLECTION
#define INFO_COLLECTION if(0)

using namespace Signal;
using namespace std;

void printUniformInfo(int program)
{
    int n_uniforms = 0;
    int len_uniform = 0;
    glGetProgramiv (program, GL_ACTIVE_UNIFORM_MAX_LENGTH, &len_uniform);
    glGetProgramiv (program, GL_ACTIVE_UNIFORMS, &n_uniforms);
    char name[len_uniform];
    Log("Found %d uniforms in program") % n_uniforms;
    for (int i=0; i<n_uniforms; i++) {
        GLint size;
        GLenum type;
        glGetActiveUniform(program,
            i,
            len_uniform,
            0,
            &size,
            &type,
            name);
        Log("%d: %s, size=%d, type=%d") % i % ((char*)name) % size % type;
    }
}

namespace Heightmap {
namespace BlockManagement {
namespace Merge {

typedef pair<Region,pBlock> RegionBlock;
typedef vector<RegionBlock> RegionBlockVector;

RegionBlockVector& operator-=(RegionBlockVector& R, Region t)
{
    RegionBlockVector C;
    C.reserve (R.size ());

    for (auto v : R) {
        Region r = v.first;
        if (r.a.time >= t.b.time || r.b.time <= t.a.time || r.a.scale >= t.b.scale || r.b.scale <= t.a.scale) {
            // keep whole
            C.push_back (v);
            continue;
        }

        if (r.a.scale < t.a.scale) {
            // keep bottom
            C.push_back (RegionBlock(Region(Position(r.a.time, r.a.scale),Position(r.b.time,t.a.scale)),v.second));
        }
        if (r.b.scale > t.b.scale) {
            // keep top
            C.push_back (RegionBlock(Region(Position(r.a.time, t.b.scale),Position(r.b.time,r.b.scale)),v.second));
        }
        if (r.a.time < t.a.time) {
            // keep left
            C.push_back (RegionBlock(Region(Position(r.a.time, max(r.a.scale,t.a.scale)),Position(t.a.time,min(r.b.scale,t.b.scale))),v.second));
        }
        if (r.b.time > t.b.time) {
            // keep right
            C.push_back (RegionBlock(Region(Position(t.b.time, max(r.a.scale,t.a.scale)),Position(r.b.time,min(r.b.scale,t.b.scale))),v.second));
        }
    }

    return R = C;
}


void testRegionBlockOperator() {
    // RegionBlockVector& operator-=(RegionBlockVector& R, Region t) should work
    try {
        Reference ref;
        ref.log2_samples_size[0] = -10;
        ref.log2_samples_size[1] = -10;
        ref.block_index[0] = 1;
        ref.block_index[1] = 1;

        BlockLayout bl(128,128,100);
        RegionFactory rf(bl);
        auto f = [&rf](Reference r) { return rf.getVisible (r); };
        RegionBlockVector V = {RegionBlock(f(ref),pBlock())};

        V -= f(ref.left ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 1u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref.right ()) );
        V -= f(ref.top ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 1u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref.right ().bottom ()) );

        V = {RegionBlock(f(ref),pBlock())};
        V -= f(ref.right ().top ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 2u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref.bottom ()) );
        EXCEPTION_ASSERT_EQUALS( V[1].first, f(ref.top ().left ()) );

        V = {RegionBlock(f(ref),pBlock())};
        V -= f(ref.sibblingLeft ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 1u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref) );
        V -= f(ref.sibblingRight ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 1u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref) );
        V -= f(ref.sibbling1 ());
        V -= f(ref.sibbling2 ());
        V -= f(ref.sibbling3 ());
        V -= f(ref.sibblingTop ());
        V -= f(ref.sibblingBottom ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 1u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref) );

        V -= f(ref.parentHorizontal ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 0u );

        V = {RegionBlock(f(ref),pBlock())};
        V -= f(ref.parentHorizontal ().top ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 1u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref.parentHorizontal ().bottom ().right ()) );
    } catch (...) {
        Log("%s: %s") % __FUNCTION__ % boost::current_exception_diagnostic_information ();
        throw;
    }
}


MergerTexture::
        MergerTexture(BlockCache::const_ptr cache, BlockLayout block_layout, bool disable_merge)
    :
      cache_(cache),
      fbo_(0),
      vbo_(0),
      tex_(0),
      block_layout_(block_layout),
      disable_merge_(disable_merge)
{
    EXCEPTION_ASSERT(cache_);
}


MergerTexture::
        ~MergerTexture()
{
    if ((vbo_ || fbo_) &&  !QOpenGLContext::currentContext ()) {
        Log ("%s: destruction without gl context leaks fbo %d and vbo %d") % __FILE__ % fbo_ % vbo_;
        return;
    }

    if (vbo_) glDeleteBuffers (1, &vbo_);
    vbo_ = 0;

    if (fbo_) glDeleteFramebuffers(1, &fbo_);
    fbo_ = 0;
}


void MergerTexture::
        init()
{
    if (vbo_)
        return;

    EXCEPTION_ASSERT(QGLContext::currentContext ());

    glGenFramebuffers(1, &fbo_);

#ifndef DRAW_STRAIGHT_ONTO_BLOCK
    // TODO if there's a point in this it should use a renderbuffer instead of a block texture
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFbo);
    tex_ = Render::BlockTextures::get1 ();
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, tex_->getOpenGlTextureId (), 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, QOpenGLContext::currentContext ()->defaultFramebufferObject ());
#endif

    glGenBuffers (1, &vbo_);
    float vertices[] = {
        0, 0, 0, 0,
        0, 1, 0, 1,
        1, 0, 1, 0,
        1, 1, 1, 1,
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //    program_ = ShaderResource::loadGLSLProgram("", ":/shaders/mergertexture.frag");
//    program_ = ShaderResource::loadGLSLProgram(":/shaders/mergertexture.vert", ":/shaders/mergertexture0.frag");
    GlException_SAFE_CALL( programp_ = ShaderResource::loadGLSLProgram(":/shaders/mergertexture.vert", ":/shaders/mergertexture.frag") );
    program_ = programp_->programId();

    qt_Vertex = glGetAttribLocation (program_, "qt_Vertex");
    qt_MultiTexCoord0 = glGetAttribLocation (program_, "qt_MultiTexCoord0");
    qt_Texture0 = glGetUniformLocation(program_, "qt_Texture0");
    invtexsize = glGetUniformLocation(program_, "invtexsize");
    uniProjection = glGetUniformLocation (program_, "qt_ProjectionMatrix");
    uniModelView = glGetUniformLocation (program_, "qt_ModelViewMatrix");
}

Signal::Intervals MergerTexture::
        fillBlocksFromOthers( const std::vector<pBlock>& blocks )
{
    GlGroupMarker gpm("MergerTexture");
    int prev_fbo = 0;
    glGetIntegerv (GL_FRAMEBUFFER_BINDING, &prev_fbo);

    init ();
    INFO_COLLECTION TaskTimer tt(boost::format("MergerTexture: fillBlocksFromOthers %s blocks") % blocks.size ());

    GlException_CHECK_ERROR();

    glClearColor (0,0,0,1);

    glViewport(0, 0, block_layout_.texels_per_row (), block_layout_.texels_per_column () );

    GlState::glDisable (GL_DEPTH_TEST, true); // disable depth test before binding framebuffer without depth buffer
    GlState::glDisable (GL_CULL_FACE);

    struct vertex_format {
        float x, y, u, v;
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);

    GlState::glEnableVertexAttribArray (qt_Vertex);
    GlState::glEnableVertexAttribArray (qt_MultiTexCoord0);

    glVertexAttribPointer (qt_Vertex, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), 0);
    glVertexAttribPointer (qt_MultiTexCoord0, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), (float*)0 + 2);

    glUseProgram (program_);
    if (invtexsize) glUniform2f(invtexsize, 1.0/block_layout_.texels_per_row (), 1.0/block_layout_.texels_per_column ());
    glUniform1i(qt_Texture0, 0); // GL_TEXTURE0 + i

    cache_clone = cache_->clone();

    Signal::Intervals I;

    {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_);

        for (pBlock b : blocks)
            I |= fillBlockFromOthersInternal (b);

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, QOpenGLContext::currentContext ()->defaultFramebufferObject ());
    }

    cache_clone.clear ();

    GlState::glDisableVertexAttribArray (qt_MultiTexCoord0);
    GlState::glDisableVertexAttribArray (qt_Vertex);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GlState::glEnable (GL_DEPTH_TEST);
    GlState::glEnable (GL_CULL_FACE);

    if (Render::BlockTextures::mipmaps > 0)
    {
        for (pBlock b : blocks)
        {
            glBindTexture (GL_TEXTURE_2D, b->texture ()->getOpenGlTextureId());
            glGenerateMipmap (GL_TEXTURE_2D);
        }
    }

    GlException_CHECK_ERROR();

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev_fbo);

    return I;
}


Signal::Intervals MergerTexture::
        fillBlockFromOthersInternal( pBlock block )
{
    Signal::Intervals missing_details;

    VERBOSE_COLLECTION TaskTimer tt(boost::format("MergerTexture: Stubbing new block %s") % block->getVisibleRegion ());

    Region r = block->getOverlappingRegion ();

    matrixd projection;
    glhOrtho (projection.v (), r.a.time, r.b.time, r.a.scale, r.b.scale, -10, 10);

    glUniformMatrix4fv (uniProjection, 1, false, GLmatrixf(projection).v ());

#ifdef DRAW_STRAIGHT_ONTO_BLOCK
    glBindTexture (GL_TEXTURE_2D, block->texture ()->getOpenGlTextureId ());
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, block->texture ()->getOpenGlTextureId (), 0);
#endif

    glClear( GL_COLOR_BUFFER_BIT );
//    clearBlock (r);

    if (!disable_merge_)
    {
        // Largest first
        RegionBlockVector largeblocks;
        RegionBlockVector smallblocks;

        for (const auto& c : cache_clone)
        {
            const pBlock& bl = c.second;
            const Region& r2 = bl->getOverlappingRegion ();

            // If r2 doesn't overlap r at all
            if (r2.a.scale >= r.b.scale || r2.b.scale <= r.a.scale )
                continue;
            if (r2.a.time >= r.b.time || r2.b.time <= r.a.time )
                continue;

            bool is_small_s = r2.scale () <= r.scale ();
            bool is_small_t = r2.time () <= r.time ();
            if (is_small_s && is_small_t)
                smallblocks.push_back (RegionBlock(r2,bl));
            else
                largeblocks.push_back (RegionBlock(r2,bl));
        }

        RegionBlockVector missing_details_region{RegionBlock(r,block)};
        for (auto r : smallblocks) {
            largeblocks -= r.first;
            missing_details_region -= r.first;
        }

        double fs = block_layout_.sample_rate ();
        for (auto v: missing_details_region)
        {
            Region r2 = v.first;
            if (r2.b.time <= 0 || r2.b.scale <= 0 || r2.a.scale >= 1)
                continue;

            double d = 0.5/v.second->sample_rate();
            missing_details |= Signal::Interval((r2.a.time-d)*fs, (r2.b.time+d)*fs+1);
        }

        for (auto v : largeblocks)
        {
            auto bl = v.second;
            mergeBlock( bl->getOverlappingRegion (), bl->texture ()->getOpenGlTextureId () );
        }

        for (auto v : smallblocks)
        {
            auto bl = v.second;
            mergeBlock( bl->getOverlappingRegion (), bl->texture ()->getOpenGlTextureId () );
        }
    }
    else
        missing_details = block->getInterval ();

#ifdef DRAW_STRAIGHT_ONTO_BLOCK
    // The texture image will not be detached if the texture is deleted but it
    // doesn't matter if the texture image is still attached.
    //   https://www.khronos.org/opengles/sdk/docs/man/xhtml/glFramebufferTexture2D.xml
    // Call this to detach:
    //   glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
    //                          GL_TEXTURE_2D, 0, 0);
#else
    {
        VERBOSE_COLLECTION TaskTimer tt(boost::format("Filled %s") % block->getOverlappingRegion ());

        GlTexture::ptr t = block->texture ();
        #ifdef GL_ES_VERSION_2_0
            GlException_SAFE_CALL( glCopyTextureLevelsAPPLE(t->getOpenGlTextureId (), fbo_->getGlTexture (), 0, 1) );
        #else
            glBindTexture(GL_TEXTURE_2D, t->getOpenGlTextureId ());
            GlException_SAFE_CALL( glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, t->getWidth (), t->getHeight ()) );
        #endif
    }
#endif

    return missing_details;
}


void MergerTexture::
        clearBlock( const Region& ri )
{
    // Read from zero texture, don't do this. The zero texture is not
    // "unbound", it's just a usually incomplete texture object and it is an
    // error to draw with an incomplete texture. Do glClear instead.
    mergeBlock(ri, 0);
}


void MergerTexture::
        mergeBlock( const Region& ri, int texture )
{
    VERBOSE_COLLECTION TaskTimer tt(boost::format("MergerTexture: Filling with %d from %s") % texture % ri);

    matrixd modelview = matrixd::identity ();
    modelview *= matrixd::translate (ri.a.time, ri.a.scale, 0);
    modelview *= matrixd::scale (ri.time (), ri.scale (), 1.f);

    glUniformMatrix4fv (uniModelView, 1, false, GLmatrixf(modelview).v ());

    glBindTexture( GL_TEXTURE_2D, texture);
    GlState::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // Paint new contents over it
}

} // namespace Merge
} // namespace BlockManagement
} // namespace Heightmap


#include "log.h"
#include "cpumemorystorage.h"
#include "heightmap/render/blocktextures.h"
#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget

namespace Heightmap {
namespace BlockManagement {
namespace Merge {

static void clearCache(BlockCache::ptr cache) {
    auto c = cache->clear ();
    c.clear();
}


void MergerTexture::
        test()
{
    std::string name = "MergerTexture";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
#ifndef LEGACY_OPENGL
    QGLFormat f = QGLFormat::defaultFormat ();
    f.setProfile( QGLFormat::CoreProfile );
    f.setVersion( 3, 2 );
    QGLFormat::setDefaultFormat (f);
#endif
    QGLWidget w;
    w.makeCurrent ();
#ifndef LEGACY_OPENGL
    GLuint VertexArrayID;
    GlException_SAFE_CALL( glGenVertexArrays(1, &VertexArrayID) );
    GlException_SAFE_CALL( glBindVertexArray(VertexArrayID) );
#endif

    testRegionBlockOperator();

    // It should merge contents from other blocks to stub the contents of a new block.
    {
        BlockCache::ptr cache(new BlockCache);

        Reference ref;
        BlockLayout bl(4,4,4);
        Render::BlockTextures::Scoped bt_raii(bl.texels_per_row (), bl.texels_per_column ());
        VisualizationParams::ptr vp(new VisualizationParams);

        // VisualizationParams has only things that have nothing to do with MergerTexture.
        pBlock block(new Block(ref,bl,vp,0));
        GlTexture::ptr tex = block->texture ();

        Log("Source overlapping %s, visible %s") % block->getOverlappingRegion () % block->getVisibleRegion ();
        MergerTexture(cache, bl).fillBlockFromOthers(block);

        DataStorage<float>::ptr data;

        float expected1[]={ 0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0};

        data = GlTextureRead(*tex).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected1, sizeof(expected1), data);

        {
            float srcdata[]={ 1, 0, 0, 0,
                              0, 2, 0, 0,
                              0, 0, 0, 0,
                              0.5, 0, 0, 0};

            pBlock block(new Block(ref.parentHorizontal (),bl,vp,0));
            //pBlock block(new Block(ref,bl,vp));
            //Log("Inserting overlapping %s, visible %s") % block->getOverlappingRegion () % block->getVisibleRegion ();
            GlTexture::ptr tex = block->texture ();
            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            cache->insert(block);
        }

        MergerTexture(cache, bl).fillBlockFromOthers(block);
        clearCache(cache);
        float expected2[]={   1,  .5,   0,   0,
                              0,   1,   2,   1,
                              0,   0,   0,   0,
                             .5, .25,   0,   0};
        data = GlTextureRead(*tex).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;
        //DataStorage<float>::ptr expected2ptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected2);
        //PRINT_DATASTORAGE(expected2ptr, "");
        //PRINT_DATASTORAGE(data, "");
        COMPARE_DATASTORAGE(expected2, sizeof(expected2), data);

        {
            float srcdata[]={ 1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 1.0, 11, 12,
                              13, 14, 15, .16};

            pBlock block(new Block(ref.right (),bl,vp,0));
            GlTexture::ptr tex = block->texture ();
            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            cache->insert(block);
        }

        MergerTexture(cache, bl).fillBlockFromOthers(block);
        float expected3[]={   0, 0,    3,  4,
                              0, 0,    7,  8,
                              0, 0,   11,  12,
                              0, 0,   15,  15};

        data = GlTextureRead(*tex).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;

        //DataStorage<float>::ptr expected3ptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected3);
        //PRINT_DATASTORAGE(expected3ptr, "");
        //PRINT_DATASTORAGE(data, "");

        COMPARE_DATASTORAGE(expected3, sizeof(expected3), data);
        clearCache(cache);

        tex.reset ();
    }
}

} // namespace Merge
} // namespace BlockManagement
} // namespace Heightmap
