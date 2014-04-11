#include "merger.h"
#include "../blockquery.h"
#include "../blockkernel.h"

#include "tasktimer.h"
#include "computationkernel.h"

#include <boost/foreach.hpp>

//#define VERBOSE_COLLECTION
#define VERBOSE_COLLECTION if(0)

//#define INFO_COLLECTION
#define INFO_COLLECTION if(0)

using namespace Signal;

namespace Heightmap {
namespace Blocks {

Merger::
        Merger(BlockCache::const_ptr cache)
    :
      cache_(cache)
{
}


void Merger::
        fillBlockFromOthers( pBlock block )
{
    VERBOSE_COLLECTION TaskTimer tt("Stubbing new block");

    const Reference& ref = block->reference ();
    Intervals things_to_update = block->getInterval ();
    std::vector<pBlock> gib = BlockQuery(cache_).getIntersectingBlocks ( things_to_update.spannedInterval(), false, 0 );
    auto outdata = block->block_data ();

    const Region& r = block->getRegion ();

    int merge_levels = 10;

    VERBOSE_COLLECTION TaskTimer tt2("Checking %u blocks out of %u blocks, %d times", gib.size(), cache_->size(), merge_levels);

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

                    mergeBlock( *block,  *bl,
                                outdata, bl->block_data_const () );
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


bool Merger::
        mergeBlock( const Block& outBlock,
                    const Block& inBlock,
                    const shared_state<BlockData>::write_ptr& poutData,
                    const shared_state<BlockData>::read_ptr& pinData )
{
    if (!poutData.get () || !pinData.get ())
        return false;

    BlockData& outData = *poutData;
    const BlockData& inData = *pinData;

    EXCEPTION_ASSERT( &outBlock != &inBlock );

    // Find out what intervals that match
    Intervals transferDesc = inBlock.getInterval() & outBlock.getInterval();

    // If the blocks doesn't overlap, there's nothing to do
    if (!transferDesc)
        return false;

    const Region& ri = inBlock.getRegion();
    const Region& ro = outBlock.getRegion();

    // If the blocks doesn't overlap, there's nothing to do
    if (ro.b.scale <= ri.a.scale || ri.b.scale<=ro.a.scale || ro.b.time <= ri.a.time || ri.b.time <= ro.a.time)
        return false;

    INFO_COLLECTION TaskTimer tt(boost::format("%s, %s into %s") % __FUNCTION__ % ri % ro);

    VERBOSE_COLLECTION ComputationSynchronize();

    // TODO examine why blockMerge is really really slow
    {
        VERBOSE_COLLECTION TaskTimer tt("blockMerge");
        ::blockMerge( inData.cpu_copy,
                      outData.cpu_copy,
                      ResampleArea( ri.a.time, ri.a.scale, ri.b.time, ri.b.scale ),
                      ResampleArea( ro.a.time, ro.a.scale, ro.b.time, ro.b.scale ) );
        VERBOSE_COLLECTION ComputationSynchronize();
    }

    //bool isCwt = dynamic_cast<const Tfr::Cwt*>(transform());
    //bool using_subtexel_aggregation = !isCwt || (renderer ? renderer->redundancy()<=1 : false);
    bool using_subtexel_aggregation = false;

#ifndef CWT_SUBTEXEL_AGGREGATION
    // subtexel aggregation is way to slow
    using_subtexel_aggregation = false;
#endif

    if (!using_subtexel_aggregation) // blockMerge doesn't support subtexel aggregation
        if (false) // disabled validation of blocks until this is stable
    {
        // Validate region of block if inBlock was source of higher resolution than outBlock
        if (inBlock.reference().log2_samples_size[0] < outBlock.reference().log2_samples_size[0] &&
            inBlock.reference().log2_samples_size[1] == outBlock.reference().log2_samples_size[1] &&
            ri.b.scale==ro.b.scale && ri.a.scale==ro.a.scale)
        {
            VERBOSE_COLLECTION TaskInfo(boost::format("Using block %s") % ReferenceInfo(
                                            inBlock.reference (),
                                            inBlock.block_layout (),
                                            inBlock.visualization_params () ));
        }
    }

    VERBOSE_COLLECTION ComputationSynchronize();

    return true;
}

} // namespace Block
} // namespace Heightmap


#include "log.h"
#include "heightmap/glblock.h"
#include "cpumemorystorage.h"

namespace Heightmap {
namespace Blocks {



// Same as in the test for ResampleTexture
static void compare(float* expected, size_t sizeof_expected, DataStorage<float>::ptr data)
{
    EXCEPTION_ASSERT(data);
    EXCEPTION_ASSERT_EQUALS(sizeof_expected, data->numberOfBytes ());

    float *p = data->getCpuMemory ();

    if (0 != memcmp(p, expected, sizeof_expected))
    {

        Log("%s") % (DataStorageSize)data->size ();
        for (size_t i=0; i<data->numberOfElements (); i++)
            Log("%s: %s\t%s\t%s") % i % p[i] % expected[i] % (p[i] - expected[i]);

        EXCEPTION_ASSERT_EQUALS(0, memcmp(p, expected, sizeof_expected));
    }
}


static void clearCache(BlockCache::ptr cache) {
    for (auto c : cache->clear ())
    {
        c.second->glblock.reset();
    }
}


void Merger::
        test()
{
    // It should merge contents from other blocks to stub the contents of a new block.
    {
        BlockCache::ptr cache(new BlockCache);

        Reference ref;
        BlockLayout bl(4,4,4);
        DataStorageSize ds(bl.texels_per_column (), bl.texels_per_row ());

        // VisualizationParams has only things that have nothing to do with MergerTexture.
        VisualizationParams::ptr vp(new VisualizationParams);
        pBlock block(new Block(ref,bl,vp));
        const Region& r = block->getRegion();
        block->glblock.reset( new GlBlock( bl, r.time(), r.scale() ));
        EXCEPTION_ASSERT_EQUALS(ds, block->glblock->heightSize());
        block->block_data()->cpu_copy.reset( new DataStorage<float>(ds) );

        Merger(cache).fillBlockFromOthers(block);
        BlockData::pData data = block->block_data ()->cpu_copy;

        float expected1[]={ 0, 0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0, 0};

        compare(expected1, sizeof(expected1), data);

        {
            float* srcdata=new float[16]{ 1, 0, 0, .5,
                                          0, 0, 0, 0,
                                          0, 0, 0, 0,
                                         .5, 0, 0, .5};

            pBlock block(new Block(ref.parentHorizontal (),bl,vp));
            const Region& r = block->getRegion();
            block->glblock.reset( new GlBlock( bl, r.time(), r.scale() ));
            block->block_data()->cpu_copy = CpuMemoryStorage::BorrowPtr( ds, srcdata, true );
            cache->insert(block);
        }

        Merger(cache).fillBlockFromOthers(block);
        clearCache(cache);
        float expected2[]={   1, 0.5,  0, 0,
                              0, 0,    0, 0,
                              0, 0,    0, 0,
                             .5, 0.25, 0, 0};
        compare(expected2, sizeof(expected2), data);

        {
            float* srcdata=new float[16]{ 1, 2, 3, 4,
                                          5, 6, 7, 8,
                                          9, 10, 11, 12,
                                          13, 14, 15, 16};

            pBlock block(new Block(ref.right (),bl,vp));
            const Region& r = block->getRegion();
            block->glblock.reset( new GlBlock( bl, r.time(), r.scale() ));
            block->block_data()->cpu_copy = CpuMemoryStorage::BorrowPtr( ds, srcdata, true );
            cache->insert(block);
        }

        Merger(cache).fillBlockFromOthers(block);
        float expected3[]={   1, 0.5,  2,  4,
                              0, 0,    6,  8,
                              0, 0,    10,  12,
                             .5, 0.25, 14, 16};
        compare(expected3, sizeof(expected3), data);
        clearCache(cache);

        block->glblock.reset ();
    }
}

} // namespace Blocks
} // namespace Heightmap

