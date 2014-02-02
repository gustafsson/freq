#include "mergertexture.h"

#include "../blockquery.h"
#include "../blockkernel.h"
#include "heightmap/glblock.h"

#include "TaskTimer.h"
#include "computationkernel.h"
#include "gltextureread.h"
#include "resampletexture.h"
#include "datastoragestring.h"

#include <boost/foreach.hpp>

//#define VERBOSE_COLLECTION
#define VERBOSE_COLLECTION if(0)

//#define INFO_COLLECTION
#define INFO_COLLECTION if(0)

using namespace Signal;

namespace Heightmap {
namespace Blocks {

MergerTexture::
        MergerTexture(BlockCache::ConstPtr cache)
    :
      cache_(cache)
{
}


void MergerTexture::
        fillBlockFromOthers( pBlock block )
{
    VERBOSE_COLLECTION TaskTimer tt("Stubbing new block");

    const Reference& ref = block->reference ();
    Intervals things_to_update = block->getInterval ();
    std::vector<pBlock> gib = BlockQuery(cache_).getIntersectingBlocks ( things_to_update.spannedInterval(), false, 0 );
    BlockData::WritePtr outdata = block->block_data ();

    const Region& r = block->getRegion ();

    ResampleTexture::Area out(r.a.time, r.a.scale, r.b.time, r.b.scale);

    GlTexture::Ptr t = block->glblock->glTexture ();
    // Must create a new texture with GL_RGBA to paint to, can't paint to a monochromatic texture
    // But only the first color component will be read from.
    // GlTexture sum(t->getWidth (), t->getHeight (), GL_RGBA, GL_RGBA, GL_FLOAT);
    GlTexture sum(t->getWidth (), t->getHeight (), GL_LUMINANCE, GL_RGBA, GL_FLOAT);
    ResampleTexture rt(&sum, out);
    rt(&*t, out); // Paint old contents into it

    int merge_levels = 10;

    VERBOSE_COLLECTION TaskTimer tt2("Checking %u blocks out of %u blocks, %d times", gib.size(), read1(cache_)->cache().size(), merge_levels);

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
                    things_to_update -= v;

                    mergeBlock( rt, *bl );
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
        } catch (const BlockData::LockFailed&) {}

        gib = next;
    }

    if (things_to_update != block->getInterval ())
      {
        TaskTimer tt(boost::format("OpenGL -> CPU -> OpenGL %s") % block->getRegion ());

        // Get it back to cpu memory
        BlockData::pData d = GlTextureRead(sum.getOpenGlTextureId ()).readFloat (0,GL_RED);

        EXCEPTION_ASSERT_EQUALS(d->size (), outdata->cpu_copy->size());
        outdata->cpu_copy = d; // replace the DataStorage, don't bother copying into the previous one

        *block->glblock->height()->data = *d;
        block->glblock->update_texture( GlBlock::GlBlock::HeightMode_Flat );
      }
}


void MergerTexture::
        mergeBlock( ResampleTexture& rt, const Block& inBlock )
{
    Region ri = inBlock.getRegion ();
    ResampleTexture::Area in(ri.a.time, ri.a.scale, ri.b.time, ri.b.scale);

    rt(inBlock.glblock->glTexture ().get (), in); // Paint new contents over it
}

} // namespace Blocks
} // namespace Heightmap


#include "log.h"
#include "heightmap/glblock.h"
#include "cpumemorystorage.h"
#include <QApplication>
#include <QGLWidget>

namespace Heightmap {
namespace Blocks {

static void clearCache(BlockCache::Ptr cache) {
    while(!read1(cache)->cache().empty()) {
        pBlock b = read1(cache)->cache().begin()->second;
        b->glblock.reset();
        write1(cache)->erase(b->reference ());
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
        BlockCache::Ptr cache(new BlockCache);

        Reference ref;
        BlockLayout bl(4,4,4);
        DataStorageSize ds(bl.texels_per_column (), bl.texels_per_row ());

        // VisualizationParams has only things that have nothing to do with MergerTexture.
        VisualizationParams::Ptr vp(new VisualizationParams);
        pBlock block(new Block(ref,bl,vp));
        const Region& r = block->getRegion();
        block->glblock.reset( new GlBlock( bl, r.time(), r.scale() ));
        EXCEPTION_ASSERT_EQUALS(ds, block->glblock->heightSize());
        block->block_data()->cpu_copy.reset( new DataStorage<float>(ds) );
        block->update_glblock_data ();
        block->glblock->update_texture( GlBlock::GlBlock::HeightMode_Flat );
        EXCEPTION_ASSERT(block->glblock->glTexture ()->getOpenGlTextureId ());

        MergerTexture(cache).fillBlockFromOthers(block);
        EXCEPTION_ASSERT(block->glblock->glTexture ()->getOpenGlTextureId ());

        BlockData::pData data = block->block_data ()->cpu_copy;
        block->update_glblock_data ();
        block->glblock->update_texture( GlBlock::GlBlock::HeightMode_Flat );

        float expected1[]={ 0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0};

        COMPARE_DATASTORAGE(expected1, sizeof(expected1), data);

        {
            float* srcdata=new float[16]{ 1, 0, 0, .5,
                                          0, 0, 0, 0,
                                          0, 0, 0, 0,
                                         .5, 0, 0, .5};

            pBlock block(new Block(ref.parentHorizontal (),bl,vp));
            const Region& r = block->getRegion();
            block->glblock.reset( new GlBlock( bl, r.time(), r.scale() ));
            block->block_data()->cpu_copy = CpuMemoryStorage::BorrowPtr( ds, srcdata, true );
            block->update_glblock_data ();
            block->glblock->update_texture( GlBlock::GlBlock::HeightMode_Flat );

            EXCEPTION_ASSERT(block->glblock->glTexture ()->getOpenGlTextureId ());
            write1(cache)->insert(block);
        }

        EXCEPTION_ASSERT(block->glblock->glTexture ()->getOpenGlTextureId ());
        MergerTexture(cache).fillBlockFromOthers(block);
        clearCache(cache);
        float a = 0.875, b = 0.75,  c = 0.25,
              d = 0.5,   e = 0.375, f = 0.125;
        float expected2[]={   a, b, c, 0,
                              0, 0, 0, 0,
                              0, 0, 0, 0,
                              d, e, f, 0};
        data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected2, sizeof(expected2), data);

        {
            float* srcdata=new float[16]{ 1, 2, 3, 4,
                                          5, 6, 7, 8,
                                          9, 10, 11, 12,
                                          13, 14, 15, .16};

            pBlock block(new Block(ref.right (),bl,vp));
            const Region& r = block->getRegion();
            block->glblock.reset( new GlBlock( bl, r.time(), r.scale() ));
            block->block_data()->cpu_copy = CpuMemoryStorage::BorrowPtr( ds, srcdata, true );
            block->update_glblock_data ();
            block->glblock->update_texture( GlBlock::GlBlock::HeightMode_Flat );

            write1(cache)->insert(block);
        }

        MergerTexture(cache).fillBlockFromOthers(block);
        float expected3[]={   a, b,    1.5,  3.5,
                              0, 0,    5.5,  7.5,
                              0, 0,    9.5,  11.5,
                              d, e,   13.5,  7.58};
        data = block->block_data ()->cpu_copy;
        COMPARE_DATASTORAGE(expected3, sizeof(expected3), data);
        clearCache(cache);

        block->glblock.reset ();
    }
}

} // namespace Blocks
} // namespace Heightmap
