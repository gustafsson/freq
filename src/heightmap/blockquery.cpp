#include "blockquery.h"
#include "tasktimer.h"

#include <numeric>

//#define INFO_COLLECTION
#define INFO_COLLECTION if(0)

using namespace std;
using namespace Signal;

namespace Heightmap {

BlockQuery::
        BlockQuery(BlockCache::const_ptr cache)
    :
      cache_(cache)
{
}


vector<pBlock> BlockQuery::
        getIntersectingBlocks( const Intervals& I, bool only_visible, int frame_counter ) const
{
    BlockCache::cache_t cache = cache_->clone();

    // Keep a lock
//    auto cache_r = cache_.read ();
//    auto cache = cache_r->cache();

    vector<pBlock> r;
    r.reserve(32);

    INFO_COLLECTION TaskTimer tt(boost::format("getIntersectingBlocks( %s, %s ) from %u caches spanning %s")
                 % I
                 % (only_visible?"only visible":"all")
                 % cache.size()
                 % accumulate(cache.begin(), cache.end(), Intervals(),
                        [&cache]( Intervals& I, const BlockCache::cache_t::value_type& c) {
                            return I |= c.second->getInterval();
                        }));

    for( auto c : cache )
    {
        const pBlock& pb = c.second;

        if (only_visible) {
            unsigned framediff = frame_counter - pb->frame_number_last_used;
            if (framediff != 0 && framediff != 1)
                continue;
        }

        if (I & pb->getInterval())
            r.push_back(pb);
    }

/*
    // consistency check
    foreach( const cache_t::value_type& c, _cache )
    {
        bool found = false;
        foreach( const recent_t::value_type& r, _recent )
        {
            if (r == c.second)
                found = true;
        }

        EXCEPTION_ASSERT(found);
    }

    // consistency check
    foreach( const recent_t::value_type& r, _recent )
    {
        bool found = false;
        foreach( const cache_t::value_type& c, _cache )
        {
            if (r == c.second)
                found = true;
        }

        EXCEPTION_ASSERT(found);
    }
*/

    return r;
}


} // namespace Heightmap
