#include "mergertexture.h"

#include "heightmap/blockquery.h"
#include "mergekernel.h"
#include "heightmap/render/glblock.h"

#include "tasktimer.h"
#include "computationkernel.h"
#include "gltextureread.h"
#include "glframebuffer.h"
#include "datastoragestring.h"
#include "GlException.h"
#include "glPushContext.h"

#include <QGLContext>

#include <boost/foreach.hpp>

//#define VERBOSE_COLLECTION
#define VERBOSE_COLLECTION if(0)

//#define INFO_COLLECTION
#define INFO_COLLECTION if(0)

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
      disable_merge_(disable_merge)
{
    EXCEPTION_ASSERT(QGLContext::currentContext ());

    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, block_layout.texels_per_row(), block_layout.texels_per_column (), 0, GL_RED, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    fbo_.reset (new GlFrameBuffer(tex_));

    glGenBuffers (1, &vbo_);
}


MergerTexture::
        ~MergerTexture()
{
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

    for (pBlock b : blocks)
        fillBlockFromOthersInternal (b);

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
}


void MergerTexture::
        fillBlockFromOthersInternal( pBlock block )
{
    INFO_COLLECTION TaskTimer tt(boost::format("MergerTexture: Stubbing new block %s") % block->getRegion ());

    const Reference& ref = block->reference ();
    Intervals things_to_update = block->getInterval ();
    std::vector<pBlock> gib = BlockQuery(cache_).getIntersectingBlocks ( things_to_update.spannedInterval(), false, 0 );

    const Region& r = block->getRegion ();

    glClear( GL_COLOR_BUFFER_BIT );
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity();
    glOrtho(r.a.time, r.b.time, r.a.scale, r.b.scale, -10,10);

    glMatrixMode ( GL_MODELVIEW );
    glLoadIdentity();

    int merge_levels = 10;

    if (!disable_merge_)
    { VERBOSE_COLLECTION TaskTimer tt2("Checking %u blocks out of %u blocks, %d times", gib.size(), cache_->size(), merge_levels);

    for (int merge_level=0; merge_level<merge_levels && things_to_update; ++merge_level)
    {
        VERBOSE_COLLECTION TaskTimer tt("%d, %s", merge_level, things_to_update.toString().c_str());

        std::vector<pBlock> next;
        BOOST_FOREACH ( const pBlock& bl, gib ) try
        {
            // The switch from high resolution blocks to low resolution blocks (or
            // the other way around) should be as invisible as possible. Therefor
            // the contents should be stubbed from whatever contents the previous
            // blocks already have, even if the contents are out-of-date.
            //
            Interval v = bl->getInterval ();

            // Check if these samples are still considered for update
            if ( (things_to_update & v).count() <= 1)
                continue;

            int d = bl->reference().log2_samples_size[0];
            d -= ref.log2_samples_size[0];
            d += bl->reference().log2_samples_size[1];
            d -= ref.log2_samples_size[1];

            if (d==merge_level || -d==merge_level)
            {
                const Region& r2 = bl->getRegion ();
                if (r2.a.scale <= r.a.scale && r2.b.scale >= r.b.scale )
                {
                    // 'bl' covers all scales in 'block' (not necessarily all time samples though)
                    if (mergeBlock( *bl ))
                        things_to_update -= v;
                }
                else if (bl->reference ().log2_samples_size[1] + 1 == ref.log2_samples_size[1])
                {
                    // mergeBlock makes sure that their scales overlap more or less
                    //mergeBlock( block, bl, 0 ); takes time
                }
                else
                {
                    // don't bother with things that doesn't match very well
                }
            }
            else
            {
                next.push_back ( bl );
            }
        } catch (const shared_state<BlockData>::lock_failed&) {}

        gib = next;
    }
    }

    {
        VERBOSE_COLLECTION TaskTimer tt(boost::format("Filled %s") % block->getRegion ());

        GlTexture::ptr t = block->glblock->glTexture ();
        glBindTexture(GL_TEXTURE_2D, t->getOpenGlTextureId ());
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, t->getWidth (), t->getHeight ());
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}


bool MergerTexture::
        mergeBlock( const Block& inBlock )
{
    if (!inBlock.glblock || !inBlock.glblock->has_texture ())
        return false;

    INFO_COLLECTION TaskTimer tt(boost::format("MergerTexture: Filling from %s") % inBlock.getRegion ());

    Region ri = inBlock.getRegion ();
    glPushMatrix ();
    glTranslatef (ri.a.time, ri.a.scale, 0);
    glScalef (ri.time (), ri.scale (), 1.f);

    glBindTexture( GL_TEXTURE_2D, inBlock.glblock->glTexture ()->getOpenGlTextureId ());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // Paint new contents over it
    glBindTexture(GL_TEXTURE_2D, 0);
    glPopMatrix ();

    return true;
}

} // namespace Merge
} // namespace BlockManagement
} // namespace Heightmap


#include "log.h"
#include "heightmap/render/glblock.h"
#include "cpumemorystorage.h"
#include "heightmap/render/blocktextures.h"
#include <QApplication>
#include <QGLWidget>

namespace Heightmap {
namespace BlockManagement {
namespace Merge {

static void clearCache(BlockCache::ptr cache) {
    for (auto c : cache->clear ())
    {
        c.second->glblock.reset();
    }
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

        Render::BlockTextures block_textures(bl);
        block_textures.setCapacityHint (2);

        // VisualizationParams has only things that have nothing to do with MergerTexture.
        pBlock block(new Block(ref,bl,vp));
        block->glblock.reset( new Render::GlBlock( block_textures.getUnusedTextures (1)[0] ));

        MergerTexture(cache, bl).fillBlockFromOthers(block);

        DataStorage<float>::ptr data;

        float expected1[]={ 0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0};

        data = GlTextureRead(block->glblock->glTexture ()->getOpenGlTextureId ()).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected1, sizeof(expected1), data);

        {
            float* srcdata=new float[16]{ 1, 0, 0, .5,
                                          0, 0, 0, 0,
                                          0, 0, 0, 0,
                                         .5, 0, 0, .5};

            pBlock block(new Block(ref.parentHorizontal (),bl,vp));
            block->glblock.reset( new Render::GlBlock( block_textures.getUnusedTextures (1)[0] ));
            block->glblock->updateTexture (srcdata, 16);

            cache->insert(block);
        }

        MergerTexture(cache, bl).fillBlockFromOthers(block);
        clearCache(cache);
        float a = 1.0, b = 0.75,  c = 0.25;
        float expected2[]={   a,   b,   c,   0,
                              0,   0,   0,   0,
                              0,   0,   0,   0,
                              a/2, b/2, c/2, 0};
        data = GlTextureRead(block->glblock->glTexture ()->getOpenGlTextureId ()).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected2, sizeof(expected2), data);

        {
            float* srcdata=new float[16]{ 1, 2, 3, 4,
                                          5, 6, 7, 8,
                                          9, 10, 11, 12,
                                          13, 14, 15, .16};

            pBlock block(new Block(ref.right (),bl,vp));
            block->glblock.reset( new Render::GlBlock( block_textures.getUnusedTextures (1)[0] ));
            block->glblock->updateTexture (srcdata,16);

            cache->insert(block);
        }

        MergerTexture(cache, bl).fillBlockFromOthers(block);
        float v16 = 7.57812476837;
        //float v32 = 7.58;
        float expected3[]={   0, 0,    1.5,  3.5,
                              0, 0,    5.5,  7.5,
                              0, 0,    9.5,  11.5,
                              0, 0,   13.5,  v16};

        data = GlTextureRead(block->glblock->glTexture ()->getOpenGlTextureId ()).readFloat (0, GL_RED);
        //data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected3, sizeof(expected3), data);
        clearCache(cache);

        block->glblock.reset ();
    }
}

} // namespace Merge
} // namespace BlockManagement
} // namespace Heightmap
