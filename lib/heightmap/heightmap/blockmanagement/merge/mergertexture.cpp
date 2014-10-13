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

#include <QGLContext>

#ifdef GL_ES_VERSION_2_0
#define DRAW_STRAIGHT_ONTO_BLOCK
#endif

//#define VERBOSE_COLLECTION
#define VERBOSE_COLLECTION if(0)

//#define INFO_COLLECTION
#define INFO_COLLECTION if(0)

using namespace Signal;

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

MergerTexture::
        MergerTexture(BlockCache::const_ptr cache, BlockLayout block_layout, bool disable_merge)
    :
      cache_(cache),
      vbo_(0),
      tex_(0),
      block_layout_(block_layout),
      disable_merge_(disable_merge),
      program_(0)
{
}


MergerTexture::
        ~MergerTexture()
{
    if (program_) glDeleteProgram(program_);
    program_ = 0;

    if (vbo_) glDeleteBuffers (1, &vbo_);
    vbo_ = 0;
}


void MergerTexture::
        init()
{
    if (vbo_)
        return;

    EXCEPTION_ASSERT(QGLContext::currentContext ());

#ifndef DRAW_STRAIGHT_ONTO_BLOCK
        tex_ = Render::BlockTextures::get1 ();
        fbo_.reset (new GlFrameBuffer(*tex_));
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
    program_ = ShaderResource::loadGLSLProgram(":/shaders/mergertexture.vert", ":/shaders/mergertexture0.frag");
}

void MergerTexture::
        fillBlocksFromOthers( const std::vector<pBlock>& blocks )
{
    init ();
    INFO_COLLECTION TaskTimer tt(boost::format("MergerTexture: fillBlocksFromOthers %s blocks") % blocks.size ());

    GlException_CHECK_ERROR();

    glClearColor (0,0,0,1);

    glViewport(0, 0, block_layout_.texels_per_row (), block_layout_.texels_per_column () );

    glDisable (GL_DEPTH_TEST);
    glDisable (GL_BLEND);
    glDisable (GL_CULL_FACE);

    struct vertex_format {
        float x, y, u, v;
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);

    int qt_Vertex = glGetAttribLocation (program_, "qt_Vertex");
    int qt_MultiTexCoord0 = glGetAttribLocation (program_, "qt_MultiTexCoord0");
    int qt_Texture0 = glGetUniformLocation(program_, "qt_Texture0");

    glEnableVertexAttribArray (qt_Vertex);
    glEnableVertexAttribArray (qt_MultiTexCoord0);
    glVertexAttribPointer (qt_Vertex, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), 0);
    glVertexAttribPointer (qt_MultiTexCoord0, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), (float*)0 + 2);

    glUseProgram (program_);
    glUniform1i(qt_Texture0, 0); // GL_TEXTURE0 + i

    cache_clone = cache_->clone();

    {
    #ifndef DRAW_STRAIGHT_ONTO_BLOCK
        auto fboBinding = fbo_->getScopeBinding ();
        (void)fboBinding; // raii
    #endif

        for (pBlock b : blocks)
            fillBlockFromOthersInternal (b);
    }

    cache_clone.clear ();

    glUseProgram (0);

    glDisableVertexAttribArray (qt_MultiTexCoord0);
    glDisableVertexAttribArray (qt_Vertex);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glEnable (GL_DEPTH_TEST);
    glEnable (GL_CULL_FACE);

    for (pBlock b : blocks)
    {
        glBindTexture (GL_TEXTURE_2D, b->texture ()->getOpenGlTextureId());
        glGenerateMipmap (GL_TEXTURE_2D);
        glBindTexture (GL_TEXTURE_2D, 0);
    }

    GlException_CHECK_ERROR();
}


void MergerTexture::
        fillBlockFromOthersInternal( pBlock block )
{
    VERBOSE_COLLECTION TaskTimer tt(boost::format("MergerTexture: Stubbing new block %s") % block->getRegion ());

    Region r = block->getRegion ();

    matrixd projection;
    glhOrtho (projection.v (), r.a.time, r.b.time, r.a.scale, r.b.scale, -10, 10);

    int uniProjection = glGetUniformLocation (program_, "qt_ProjectionMatrix");
    glUniformMatrix4fv (uniProjection, 1, false, GLmatrixf(projection).v ());

#ifdef DRAW_STRAIGHT_ONTO_BLOCK
    GlFrameBuffer fbo(*block->texture ());
    auto fboBinding = fbo.getScopeBinding ();
    (void)fboBinding; // raii
#endif

    glClear( GL_COLOR_BUFFER_BIT );

    if (!disable_merge_)
    {
        class isRegionLarger{
        public:
            bool operator()(const pBlock& a, const pBlock& b) const {
                const Region ra = a->getRegion ();
                const Region rb = b->getRegion ();
                float va = ra.time ()*ra.scale ();
                float vb = rb.time ()*rb.scale ();
                if (va != vb)
                    return va > vb;
                if (ra.time () != rb.time())
                    return ra.time () > rb.time();
                if (ra.scale () != rb.scale())
                    return ra.scale () > rb.scale();

                // They have the same size, order by position
                if (ra.a.time != rb.a.time)
                    return ra.a.time > rb.a.time;
                return ra.a.scale > rb.a.scale;
            }
        };

        // Largest first
        std::set<pBlock, isRegionLarger> tomerge;
        pBlock smallest_larger;

        for( const auto& c : cache_clone )
        {
            const pBlock& bl = c.second;
            unsigned age = block->frame_number_last_used - bl->frame_number_last_used;
            if (age > 1)  {
                // Don't bother merging with blocks that aren't up-to-date
                continue;
            }

            const Region& r2 = bl->getRegion ();
            // If r2 doesn't overlap r at all
            if (r2.a.scale >= r.b.scale || r2.b.scale <= r.a.scale )
                continue;
            if (r2.a.time >= r.b.time || r2.b.time <= r.a.time )
                continue;

            // If r2 covers all of r
            if (r2.a.scale <= r.a.scale && r2.b.scale >= r.b.scale && r2.a.time <= r.a.time && r2.b.time >= r.b.time)
            {
                if (!smallest_larger || isRegionLarger()(smallest_larger, bl))
                    smallest_larger = bl;
            }
            // If r2 has the same scale extent (but more time details, would be smallest_larger otherwise)
            else if (r2.a.scale == r.a.scale && r2.b.scale == r.b.scale)
                tomerge.insert (bl);
            // If r2 has the same time extent (but more scale details, would be smallest_larger otherwise)
            else if (r2.a.time == r.a.time && r2.b.time == r.b.time)
                tomerge.insert (bl);
                // If r covers all of r2
            else if (r.a.scale <= r2.a.scale && r.b.scale >= r2.b.scale && r.a.time <= r2.a.time && r.b.time >= r2.b.time)
                tomerge.insert (bl);
        }

        if (smallest_larger)
            tomerge.insert (smallest_larger);

        // TODO filter 'smaller' in a smart way to remove blocks that doesn't contribute

        // Merge everything in order from largest to smallest
        for( pBlock bl : tomerge )
            mergeBlock( bl->getRegion (), bl->texture ()->getOpenGlTextureId () );
    }

#ifndef DRAW_STRAIGHT_ONTO_BLOCK
    {
        VERBOSE_COLLECTION TaskTimer tt(boost::format("Filled %s") % block->getRegion ());

        GlTexture::ptr t = block->texture ();
        #ifdef GL_ES_VERSION_2_0
            GlException_SAFE_CALL( glCopyTextureLevelsAPPLE(t->getOpenGlTextureId (), fbo_->getGlTexture (), 0, 1) );
        #else
            glBindTexture(GL_TEXTURE_2D, t->getOpenGlTextureId ());
            GlException_SAFE_CALL( glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, t->getWidth (), t->getHeight ()) );
            glBindTexture(GL_TEXTURE_2D, 0);
        #endif
    }
#endif
}


void MergerTexture::
        clearBlock( const Region& ri )
{
    // Read from unbound texture, i.e paint zero
    mergeBlock(ri,0);
}


void MergerTexture::
        mergeBlock( const Region& ri, int texture )
{
    VERBOSE_COLLECTION TaskTimer tt(boost::format("MergerTexture: Filling %d from %s") % texture % ri);

    matrixd modelview = matrixd::identity ();
    modelview *= matrixd::translate (ri.a.time, ri.a.scale, 0);
    modelview *= matrixd::scale (ri.time (), ri.scale (), 1.f);

    int uniModelView = glGetUniformLocation (program_, "qt_ModelViewMatrix");
    glUniformMatrix4fv (uniModelView, 1, false, GLmatrixf(modelview).v ());

    glBindTexture( GL_TEXTURE_2D, texture);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // Paint new contents over it
    glBindTexture(GL_TEXTURE_2D, 0);
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
    QGLWidget w;
    w.makeCurrent ();

    // It should merge contents from other blocks to stub the contents of a new block.
    {
        BlockCache::ptr cache(new BlockCache);

        Reference ref;
        BlockLayout bl(4,4,4);
        Render::BlockTextures::Scoped bt_raii(bl.texels_per_row (), bl.texels_per_column ());
        VisualizationParams::ptr vp(new VisualizationParams);

        // VisualizationParams has only things that have nothing to do with MergerTexture.
        pBlock block(new Block(ref,bl,vp));
        GlTexture::ptr tex = block->texture ();

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
            float srcdata[]={ 1, 0, 0, .5,
                              0, 0, 0, 0,
                              0, 0, 0, 0,
                             .5, 0, 0, .5};

            pBlock block(new Block(ref.parentHorizontal (),bl,vp));
            GlTexture::ptr tex = block->texture ();
            auto ts = tex->getScopeBinding ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            cache->insert(block);
            (void)ts;
        }

        MergerTexture(cache, bl).fillBlockFromOthers(block);
        clearCache(cache);
        float a = 1.0, b = 0.75,  c = 0.25;
        float expected2[]={   a,   b,   c,   0,
                              0,   0,   0,   0,
                              0,   0,   0,   0,
                              a/2, b/2, c/2, 0};
        data = GlTextureRead(*tex).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected2, sizeof(expected2), data);

        {
            float srcdata[]={ 1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 10, 11, 12,
                              13, 14, 15, .16};

            pBlock block(new Block(ref.right (),bl,vp));
            GlTexture::ptr tex = block->texture ();
            auto ts = tex->getScopeBinding ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            cache->insert(block);
            (void)ts;
        }

        MergerTexture(cache, bl).fillBlockFromOthers(block);
        float v16 = 7.57812476837;
        //float v32 = 7.58;
        float expected3[]={   0, 0,    1.5,  3.5,
                              0, 0,    5.5,  7.5,
                              0, 0,    9.5,  11.5,
                              0, 0,   13.5,  v16};

        data = GlTextureRead(*tex).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected3, sizeof(expected3), data);
        clearCache(cache);

        tex.reset ();
    }
}

} // namespace Merge
} // namespace BlockManagement
} // namespace Heightmap
