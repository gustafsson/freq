#include "clearinterval.h"
#include "heightmap/blockkernel.h"

#include <boost/foreach.hpp>

namespace Heightmap {
namespace Blocks {

ClearInterval::
        ClearInterval(BlockCache::Ptr cache)
    :
      cache_(cache)
{
}


std::list<pBlock> ClearInterval::
        discardOutside(Signal::Interval& I)
{
    BlockCache::WritePtr cache(cache_);

    std::list<pBlock> discarded;

    BlockCache::cache_t C = read1(cache_)->cache();
    BOOST_FOREACH(BlockCache::cache_t::value_type itr, cache->cache ())
    {
        pBlock block(itr.second);
        Signal::Interval blockInterval = block->getInterval();
        Signal::Interval toKeep = I & blockInterval;
        bool remove_entire_block = toKeep == Signal::Interval();
        bool keep_entire_block = toKeep == blockInterval;
        if ( remove_entire_block )
        {
            discarded.push_back (block);
        }
        else if ( keep_entire_block )
        {
        }
        else
        {
            // clear partial block
            if( I.first <= blockInterval.first && I.last < blockInterval.last )
            {
                Region ir = block->getRegion ();
                float t = I.last / block->block_layout ().targetSampleRate() - ir.a.time;

                BlockData::WritePtr bd(block->block_data());

                ::blockClearPart( bd->cpu_copy,
                              ceil(t * block->sample_rate ()) );

                block->new_data_available = true;
            }
        }
    }

    return discarded;
}

} // namespace Blocks
} // namespace Heightmap
