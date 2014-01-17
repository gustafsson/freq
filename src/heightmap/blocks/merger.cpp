#include "merger.h"
#include "../blockquery.h"
#include "../blockkernel.h"

#include "TaskTimer.h"
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
        Merger(BlockCache::ConstPtr cache)
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
    BlockData::WritePtr outdata = block->block_data ();

    const Region& r = block->getRegion ();

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
        } catch (const BlockData::LockFailed&) {}

        gib = next;
    }
}


bool Merger::
        mergeBlock( const Block& outBlock, const Block& inBlock, const BlockData::WritePtr& poutData, const BlockData::ReadPtr& pinData )
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

    INFO_COLLECTION ComputationSynchronize();

    // TODO examine why blockMerge is really really slow
    {
        INFO_COLLECTION TaskTimer tt("blockMerge");
        ::blockMerge( inData.cpu_copy,
                      outData.cpu_copy,
                      ResampleArea( ri.a.time, ri.a.scale, ri.b.time, ri.b.scale ),
                      ResampleArea( ro.a.time, ro.a.scale, ro.b.time, ro.b.scale ) );
        INFO_COLLECTION ComputationSynchronize();
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
