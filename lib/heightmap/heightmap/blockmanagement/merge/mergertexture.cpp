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
    string name(len_uniform+1, '\0');
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
            &name[0]);
        Log("%d: %s, size=%d, type=%d") % i % name % size % type;
    }
}

namespace Heightmap {
namespace BlockManagement {
namespace Merge {

typedef pair<Region,Block const*> RegionBlock;
typedef vector<RegionBlock> RegionBlockVector;

RegionBlockVector operator-(RegionBlockVector const& R, Region t)
{
    RegionBlockVector C;
    C.reserve (R.size ()+8);

    for (const auto& v : R) {
        Region const& r = v.first;

        if (r.a.time >= t.b.time || r.b.time <= t.a.time || r.a.scale >= t.b.scale || r.b.scale <= t.a.scale) {
            // keep whole
            C.push_back (v);
            continue;
        }

        // There is an interception
        if (r.a.scale < t.a.scale) {
            // keep bottom, possibly including left and right
            C.push_back (RegionBlock(Region(Position(r.a.time, r.a.scale),Position(r.b.time,t.a.scale)),v.second));
        }
        if (r.b.scale > t.b.scale) {
            // keep top, possibly including left and right
            C.push_back (RegionBlock(Region(Position(r.a.time, t.b.scale),Position(r.b.time,r.b.scale)),v.second));
        }
        if (r.a.time < t.a.time) {
            // keep left, strictly left of, but not above or below
            C.push_back (RegionBlock(Region(Position(r.a.time, max(r.a.scale,t.a.scale)),Position(t.a.time,min(r.b.scale,t.b.scale))),v.second));
        }
        if (r.b.time > t.b.time) {
            // keep right, strictly right of, but not above or below
            C.push_back (RegionBlock(Region(Position(t.b.time, max(r.a.scale,t.a.scale)),Position(r.b.time,min(r.b.scale,t.b.scale))),v.second));
        }
    }

    return C;
}

RegionBlockVector& operator-=(RegionBlockVector& R, Region t)
{
    (R-t).swap(R);
    return R;
}

Region operator&(const Region& x, const Region& y) {
    Region r { Position(std::max(x.a.time, y.a.time), std::max(x.a.scale, y.a.scale)),
               Position(std::min(x.b.time, y.b.time), std::min(x.b.scale, y.b.scale)) };
    r.b.time = std::max(r.a.time, r.b.time);
    r.b.scale = std::max(r.a.scale, r.b.scale);
    return r;
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
        RegionBlockVector V = {RegionBlock(f(ref),nullptr)};

        V -= f(ref.left ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 1u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref.right ()) );
        V -= f(ref.top ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 1u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref.right ().bottom ()) );

        V = {RegionBlock(f(ref),nullptr)};
        V -= f(ref.right ().top ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 2u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref.bottom ()) );
        EXCEPTION_ASSERT_EQUALS( V[1].first, f(ref.top ().left ()) );

        V = {RegionBlock(f(ref),nullptr)};
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

        V = {RegionBlock(f(ref),nullptr)};
        V -= f(ref.parentHorizontal ().top ());
        EXCEPTION_ASSERT_EQUALS( V.size(), 1u );
        EXCEPTION_ASSERT_EQUALS( V[0].first, f(ref.parentHorizontal ().bottom ().right ()) );
    } catch (...) {
        Log("%s: %s") % __FUNCTION__ % boost::current_exception_diagnostic_information ();
        throw;
    }
}


MergerTexture::
        MergerTexture(BlockCache::const_ptr cache, BlockLayout block_layout, int quality)
    :
      cache_(cache),
      fbo_(0),
      vbo_(0),
      tex_(0),
      block_layout_(block_layout),
      quality_(quality)
{
    EXCEPTION_ASSERT(cache_);
    EXCEPTION_ASSERT_LESS_OR_EQUAL(0, quality);
    EXCEPTION_ASSERT_LESS_OR_EQUAL(quality,2);
}


MergerTexture::
        ~MergerTexture()
{
    if ((vbo_ || fbo_) &&  !QOpenGLContext::currentContext ()) {
        Log ("%s: destruction without gl context leaks fbo %d and vbo %d") % __FILE__ % fbo_ % vbo_;
        return;
    }

    if (vbo_) GlState::glDeleteBuffers (1, &vbo_);
    vbo_ = 0;

    if (fbo_) glDeleteFramebuffers(1, &fbo_);
    fbo_ = 0;
}


void MergerTexture::
        init()
{
    if (vbo_)
        return;

    EXCEPTION_ASSERT(QOpenGLContext::currentContext ());

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

    GlState::glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STREAM_DRAW);
    GlState::glBindBuffer(GL_ARRAY_BUFFER, 0);

    switch(quality_)
    {
    case 0:
        GlException_SAFE_CALL( programp_ = ShaderResource::loadGLSLProgram(":/shaders/mergertexture.vert", ":/shaders/mergertexture0.frag") );
        break;
    case 1:
        GlException_SAFE_CALL( programp_ = ShaderResource::loadGLSLProgram(":/shaders/mergertexture.vert", ":/shaders/mergertexture1.frag") );
        break;
    case 2:
        GlException_SAFE_CALL( programp_ = ShaderResource::loadGLSLProgram(":/shaders/mergertexture.vert", ":/shaders/mergertexture.frag") );
        break;
    default:
        EXCEPTION_ASSERTX(false, boost::format("Unknown quality: %d") % quality_);
    }

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

    GlState::glBindBuffer(GL_ARRAY_BUFFER, vbo_);

    GlState::glEnableVertexAttribArray (qt_Vertex);
    GlState::glEnableVertexAttribArray (qt_MultiTexCoord0);

    glVertexAttribPointer (qt_Vertex, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), 0);
    glVertexAttribPointer (qt_MultiTexCoord0, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), (float*)0 + 2);

    GlState::glUseProgram (program_);
    glUniform2f(invtexsize, 1.0/block_layout_.texels_per_row (), 1.0/block_layout_.texels_per_column ());
    glUniform1i(qt_Texture0, 0); // GL_TEXTURE0 + i

    cache_clone = cache_->clone();

    Signal::Intervals I;

    {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_);

        for (const pBlock& b : blocks)
            I |= fillBlockFromOthersInternal (b.get ());

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, QOpenGLContext::currentContext ()->defaultFramebufferObject ());
    }

    bool has_ota = false;
    if (!cache_clone.empty ())
        has_ota = cache_clone.begin ()->second->texture_ota();
    cache_clone.clear ();

    GlState::glDisableVertexAttribArray (qt_MultiTexCoord0);
    GlState::glDisableVertexAttribArray (qt_Vertex);
    GlState::glBindBuffer(GL_ARRAY_BUFFER, 0);

    GlState::glEnable (GL_DEPTH_TEST);
    GlState::glEnable (GL_CULL_FACE);

    for (const pBlock& b : blocks)
    {
        b->enableOta (has_ota);
        b->generateMipmap ();
    }

    GlException_CHECK_ERROR();

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev_fbo);

    return I;
}


Signal::Intervals MergerTexture::
        fillBlockFromOthersInternal( const Block* out_block )
{
    Signal::Intervals missing_details;

    VERBOSE_COLLECTION TaskTimer tt(boost::format("MergerTexture: Stubbing new block %s") % out_block->getVisibleRegion ());

    Region out_r = out_block->getOverlappingRegion ();

    matrixd projection;
    glhOrtho (projection.v (), out_r.a.time, out_r.b.time, out_r.a.scale, out_r.b.scale, -10, 10);

    glUniformMatrix4fv (uniProjection, 1, false, GLmatrixf(projection).v ());

#ifdef DRAW_STRAIGHT_ONTO_BLOCK
    glBindTexture (GL_TEXTURE_2D, out_block->texture ()->getOpenGlTextureId ());
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, out_block->texture ()->getOpenGlTextureId (), 0);
#endif

    glClear( GL_COLOR_BUFFER_BIT );
//    clearBlock (r);

    bool disable_merge = false;
    if (!disable_merge)
    {
        // Largest first
        RegionBlockVector largeblocks;
        RegionBlockVector smallblocks;
        smallblocks.reserve (cache_clone.size ());

        for (auto const& c : cache_clone)
        {
            Block* const& bl = c.second.get ();
            Region const& in_r = bl->getVisibleRegion ();
            Region r = in_r & out_r;

            // If r2 doesn't overlap r at all
            if (r.scale () * r.time () == 0)
                continue;

            bool is_small_s = in_r.scale () <= out_r.scale ();
            bool is_small_t = in_r.time () <= out_r.time ();
            if (is_small_s && is_small_t)
                smallblocks.push_back (RegionBlock(r,bl));
            else
                largeblocks.push_back (RegionBlock(r,bl));
        }

        // sort smallblocks by covered area, descendingly. I.e blocks that are least useful area last
        std::sort(smallblocks.begin (), smallblocks.end (),
                  [out_r](const RegionBlockVector::value_type& a,const RegionBlockVector::value_type& b)
        {
            return a.first.time ()*a.first.scale () > b.first.time ()*b.first.scale ();
        });

        // remove redundant smallblocks, walk backwards, most likely to have redudant blocks by the end
        for (auto i = smallblocks.end (); i!=smallblocks.begin ();)
        {
            i--;

            RegionBlockVector V{RegionBlock(i->first,nullptr)};
            // is r already covered?

            for (auto j = smallblocks.begin (); j!=smallblocks.end (); j++)
            {
                if (i==j)
                    continue;
                V -= j->first;
            }

            if (V.empty ())
                i = smallblocks.erase (i);
            if (V.size ()==1)
                i->first = V[0].first;
        }

        // figure out which regions are not covered by any smallblock
        RegionBlockVector missing_details_region{RegionBlock(out_r,out_block)};
        for (const auto& v : smallblocks) {
            largeblocks -= v.first;
            missing_details_region -= v.first;
        }

        double fs = block_layout_.sample_rate ();
        for (const auto& v: missing_details_region)
        {
            const Region& r2 = v.first;
            if (r2.b.time <= 0 || r2.b.scale <= 0 || r2.a.scale >= 1)
                continue;

            double d = 0.5/v.second->sample_rate();
            missing_details |= Signal::Interval((r2.a.time-d)*fs, (r2.b.time+d)*fs+1);
        }

        // remove duplicate consecutive blocks in largeblocks left by operator-=
        std::unique(largeblocks.begin (), largeblocks.end (), [](const RegionBlockVector::value_type& a,const RegionBlockVector::value_type& b){return a.second == b.second;});
        for (const auto& v : largeblocks)
        {
            const auto& bl = v.second;
            mergeBlock( bl->getOverlappingRegion (), bl->texture ()->getOpenGlTextureId () );
        }

        for (const auto& v : smallblocks)
        {
            const auto& bl = v.second;
            mergeBlock( bl->getOverlappingRegion (), bl->texture ()->getOpenGlTextureId () );
        }

        // mergertexture is buggy, but it's still better than nothing
        // set "missing_details = block->getInterval ()" to recompute the results
    }
    else
        missing_details = out_block->getInterval ();

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
    EXCEPTION_ASSERT_NOTEQUALS(texture,0);

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
#include "heightmap/blockmanagement/mipmapbuilder.h"
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
    GlState::assume_default_gl_states ();

    testRegionBlockOperator();

    // It should merge contents from other blocks to stub the contents of a new block.
    {
        BlockCache::ptr cache(new BlockCache);

        Reference ref;
        BlockLayout bl(4,4,4);
        Render::BlockTextures::Scoped bt_raii(bl.texels_per_row (), bl.texels_per_column ());
        VisualizationParams::ptr vp(new VisualizationParams);

        // VisualizationParams has only things that have nothing to do with MergerTexture.
        pBlock target_block(new Block(ref,bl,vp,0));
        GlTexture::ptr target_tex = target_block->texture ();

        //Log("Source overlapping %s, visible %s") % target_block->getOverlappingRegion () % target_block->getVisibleRegion ();
        MergerTexture(cache, bl).fillBlockFromOthers(target_block);

        DataStorage<float>::ptr data;

        {
            float expected[]={  0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0};

            data = GlTextureRead(*target_tex).readFloat (0, GL_RED);
            COMPARE_DATASTORAGE(expected, sizeof(expected), data);
        }

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
            MergerTexture(cache, bl, 2).fillBlockFromOthers(target_block);
            clearCache(cache);

            float expected[]={    1,  .5,   0,   0,
                                  0,   1,   2,   1,
                                  0,   0,   0,   0,
                                  .5, .25,   0,   0};
            data = GlTextureRead(*target_tex).readFloat (0, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(expected, sizeof(expected), data);
        }

        {
            float srcdata[]={ 1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 1.0, 11, 12,
                              13, 14, 15, .16};

            pBlock block(new Block(ref.right (),bl,vp,0));
            GlTexture::ptr tex = block->texture ();
            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );
            tex->setMinFilter (GL_NEAREST_MIPMAP_NEAREST);
            MipmapBuilder().generateMipmap (*tex, MipmapBuilder::MipmapOperator_Max);

            cache->insert(block);
            MergerTexture(cache, bl, 2).fillBlockFromOthers(target_block);
            clearCache(cache);

            float expected[]={    0, 0,    3,  4,
                                  0, 0,    7,  8,
                                  0, 0,   11,  12,
                                  0, 0,   15,  15};

            data = GlTextureRead(*target_tex).readFloat (0, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected3);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(expected, sizeof(expected), data);
        }

        {
            float srcdata[]={ 11, 13, 12, 10,
                              9, 8, 7, 6,
                              1, 2, 3, 4,
                              15, 14, 13, 16};

            pBlock block(new Block(ref.right (),bl,vp,0));
            GlTexture::ptr tex = block->texture ();
            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            cache->insert(block);
            MergerTexture(cache, bl, 2).fillBlockFromOthers(target_block);
            clearCache(cache);

            float expected[]={    0, 0,   13,  12,
                                  0, 0,    9,  7,
                                  0, 0,    3,  4,
                                  0, 0,   15,  16};

            data = GlTextureRead(*target_tex).readFloat (0, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected4);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(expected, sizeof(expected), data);
        }

        {
            float srcdata[]={ 11, 13, 12, 10,
                              9, 8, 7, 6,
                              1, 2, 3, 4,
                              15, 14, 13, 16};

            pBlock block(new Block(ref.left (),bl,vp,0));
            GlTexture::ptr tex = block->texture ();
            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            cache->insert(block);
            MergerTexture(cache, bl, 2).fillBlockFromOthers(target_block);
            clearCache(cache);

            float expected[]={    13, 13,  0,  0,
                                  9,  8,   0,  0,
                                  2,  4,   0,  0,
                                  15, 16,  0,  0};

            data = GlTextureRead(*target_tex).readFloat (0, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(expected, sizeof(expected), data);
        }

        {
            float srcdata[]={ 11, 13, 12, 10,
                              9, 8, 7, 6,
                              1, 2, 3, 4,
                              15, 14, 13, 16};

            pBlock block(new Block(ref.bottom (),bl,vp,0));
            GlTexture::ptr tex = block->texture ();
            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            cache->insert(block);
            MergerTexture(cache, bl, 2).fillBlockFromOthers(target_block);
            clearCache(cache);

            float expected[]={    11, 13,  12, 10,
                                  15, 14,  13, 16,
                                  0,  0,   0,  0,
                                  0,  0,   0,  0};

            data = GlTextureRead(*target_tex).readFloat (0, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(expected, sizeof(expected), data);
        }

        {
            float srcdata[]={ 11, 13, 12, 10,
                              9, 8, 7, 6,
                              1, 2, 3, 4,
                              15, 14, 13, 16};

            pBlock block(new Block(ref.top (),bl,vp,0));
            GlTexture::ptr tex = block->texture ();
            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            cache->insert(block);
            MergerTexture(cache, bl, 2).fillBlockFromOthers(target_block);
            clearCache(cache);

            float expected[]={   0,   0,   0,   0,
                                 0,   0,   0,   0,
                                 11,  13,  12,  10,
                                 15,  14,  13,  16};

            data = GlTextureRead(*target_tex).readFloat (0, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(expected, sizeof(expected), data);
        }

        target_tex.reset ();
    }
}

} // namespace Merge
} // namespace BlockManagement
} // namespace Heightmap
