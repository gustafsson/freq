#include "mergertexture.h"

#include "heightmap/blockquery.h"
#include "mergekernel.h"
#include "heightmap/render/shaderresource.h"

#include "tasktimer.h"
#include "computationkernel.h"
#include "gltextureread.h"
#include "glframebuffer.h"
#include "datastoragestring.h"
#include "GlException.h"
#include "glPushContext.h"

#include <QGLContext>

#define VERBOSE_COLLECTION
//#define VERBOSE_COLLECTION if(0)

#define INFO_COLLECTION
//#define INFO_COLLECTION if(0)

using namespace Signal;

namespace Heightmap {
namespace BlockManagement {
namespace Merge {

MergerTexture::
        MergerTexture(BlockCache::const_ptr cache, BlockLayout block_layout, bool disable_merge)
    :
      cache_(cache),
      block_layout_(block_layout),
      tex_(0),
      disable_merge_(disable_merge),
      program_(0)
{
    EXCEPTION_ASSERT(QGLContext::currentContext ());

    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, block_layout.texels_per_row(), block_layout.texels_per_column (), 0, GL_RED, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    fbo_.reset (new GlFrameBuffer(tex_));

    glGenBuffers (1, &vbo_);

//    program_ = ShaderResource::loadGLSLProgram("", ":/shaders/mergertexture.frag");
}


MergerTexture::
        ~MergerTexture()
{
    if (program_)
        glDeleteProgram(program_);
    program_ = 0;

    glDeleteTextures (1, &tex_);
    tex_ = 0;

    glDeleteBuffers (1, &vbo_);
    vbo_ = 0;
}


void MergerTexture::
        fillBlocksFromOthers( const std::vector<pBlock>& blocks )
{
    GlException_CHECK_ERROR();

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    float v[4];
    glGetFloatv(GL_COLOR_CLEAR_VALUE, v);
    glClearColor (0,0,0,1);

    auto fboBinding = fbo_->getScopeBinding ();

    glViewport(0, 0, block_layout_.texels_per_row (), block_layout_.texels_per_column () );
    glMatrixMode (GL_PROJECTION);
    glPushMatrix ();
    glMatrixMode (GL_MODELVIEW);
    glPushMatrix ();

    glPushAttribContext pa(GL_ENABLE_BIT);
    glDisable (GL_DEPTH_TEST);
    glDisable (GL_BLEND);
    glDisable (GL_CULL_FACE);
    glEnable (GL_TEXTURE_2D);

    struct vertex_format {
        float x, y, u, v;
    };

    float vertices[] = {
        0, 0, 0, 0,
        0, 1, 0, 1,
        1, 0, 1, 0,
        1, 1, 1, 1,
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STREAM_DRAW);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glVertexPointer(2, GL_FLOAT, sizeof(vertex_format), 0);
    glTexCoordPointer(2, GL_FLOAT, sizeof(vertex_format), (float*)0 + 2);

    glUseProgram (program_);

    cache_clone = cache_->clone();

    for (pBlock b : blocks)
        fillBlockFromOthersInternal (b);

    cache_clone.clear ();

    glUseProgram (0);

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glMatrixMode (GL_PROJECTION);
    glPopMatrix ();
    glMatrixMode (GL_MODELVIEW);
    glPopMatrix ();

    glViewport(viewport[0], viewport[1], viewport[2], viewport[3] );
    glClearColor (v[0],v[1],v[2],v[3]);

    (void)fboBinding; // RAII
    GlException_CHECK_ERROR();

    // Flush all these drawing commands onto the device queue before the
    // updater starts issuing drawing commands onto the device.
    glFlush ();
}


void MergerTexture::
        fillBlockFromOthersInternal( pBlock block )
{
    INFO_COLLECTION TaskTimer tt(boost::format("MergerTexture: Stubbing new block %s") % block->getRegion ());

    Region r = block->getRegion ();

    glClear( GL_COLOR_BUFFER_BIT );

    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    glOrtho(r.a.time, r.b.time, r.a.scale, r.b.scale, -10,10);
    glMatrixMode ( GL_MODELVIEW );

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
            mergeBlock( *bl );
    }

    {
        VERBOSE_COLLECTION TaskTimer tt(boost::format("Filled %s") % block->getRegion ());

        GlTexture::ptr t = block->texture ();
        glBindTexture(GL_TEXTURE_2D, t->getOpenGlTextureId ());
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, t->getWidth (), t->getHeight ());
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}


bool MergerTexture::
        mergeBlock( const Block& inBlock )
{
    VERBOSE_COLLECTION TaskTimer tt(boost::format("MergerTexture: Filling from %s") % inBlock.getRegion ());

    Region ri = inBlock.getRegion ();
    glLoadIdentity();
    glTranslatef (ri.a.time, ri.a.scale, 0);
    glScalef (ri.time (), ri.scale (), 1.f);

    glBindTexture( GL_TEXTURE_2D, inBlock.texture ()->getOpenGlTextureId ());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // Paint new contents over it
    glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}

} // namespace Merge
} // namespace BlockManagement
} // namespace Heightmap


#include "log.h"
#include "cpumemorystorage.h"
#include "heightmap/render/blocktextures.h"
#include <QApplication>
#include <QGLWidget>

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
        VisualizationParams::ptr vp(new VisualizationParams);

        Render::BlockTextures block_textures(4,4);

        // VisualizationParams has only things that have nothing to do with MergerTexture.
        GlTexture::ptr tex = block_textures.get1 ();
        pBlock block(new Block(ref,bl,vp,tex));

        MergerTexture(cache, bl).fillBlockFromOthers(block);

        DataStorage<float>::ptr data;

        float expected1[]={ 0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0};

        data = GlTextureRead(tex->getOpenGlTextureId ()).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected1, sizeof(expected1), data);

        {
            float srcdata[]={ 1, 0, 0, .5,
                              0, 0, 0, 0,
                              0, 0, 0, 0,
                             .5, 0, 0, .5};

            GlTexture::ptr tex = block_textures.get1 ();
            pBlock block(new Block(ref.parentHorizontal (),bl,vp,tex));
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
        data = GlTextureRead(tex->getOpenGlTextureId ()).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected2, sizeof(expected2), data);

        {
            float srcdata[]={ 1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 10, 11, 12,
                              13, 14, 15, .16};

            GlTexture::ptr tex = block_textures.get1 ();
            pBlock block(new Block(ref.right (),bl,vp,tex));
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

        data = GlTextureRead(tex->getOpenGlTextureId ()).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected3, sizeof(expected3), data);
        clearCache(cache);

        tex.reset ();
    }
}

} // namespace Merge
} // namespace BlockManagement
} // namespace Heightmap
