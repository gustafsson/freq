#include "clearinterval.h"
#include "heightmap/update/blockkernel.h"

#include <boost/foreach.hpp>

namespace Heightmap {
namespace Blocks {

ClearInterval::
        ClearInterval(BlockCache::ptr cache)
    :
      cache_(cache)
{
}


std::list<pBlock> ClearInterval::
        discardOutside(Signal::Interval& I)
{
    std::list<pBlock> discarded;

    BlockCache::cache_t C = cache_->clone();
    BOOST_FOREACH(BlockCache::cache_t::value_type itr, C)
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

                auto bd = block->block_data();

                ::blockClearPart( bd->cpu_copy,
                              ceil(t * block->sample_rate ()) );
            }
        }
    }

    return discarded;
}

} // namespace Blocks
} // namespace Heightmap
