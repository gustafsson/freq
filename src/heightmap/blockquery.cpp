#include "blockquery.h"
#include "tasktimer.h"

#include <boost/foreach.hpp>

//#define INFO_COLLECTION
#define INFO_COLLECTION if(0)

using namespace Signal;

namespace Heightmap {

BlockQuery::
        BlockQuery(BlockCache::ConstPtr cache)
    :
      cache_(cache)
{
}


std::vector<pBlock> BlockQuery::
        getIntersectingBlocks( const Intervals& I, bool only_visible, int frame_counter ) const
{
    auto cache = cache_.read ();
    std::vector<pBlock> r;
    r.reserve(32);

    INFO_COLLECTION TaskTimer tt(boost::format("getIntersectingBlocks( %s, %s ) from %u caches spanning %s")
                 % I
                 % (only_visible?"only visible":"all")
                 % cache->cache().size()
                 % [&]()
                 {
                    Intervals J;
                    BOOST_FOREACH( const BlockCache::cache_t::value_type& c, cache->cache ())
                    {
                        const pBlock& pb = c.second;
                        J |= pb->getInterval();
                    }
                    return J;
                }());


    BOOST_FOREACH( const BlockCache::cache_t::value_type& c, cache->cache() )
    {
        const pBlock& pb = c.second;

        if (only_visible) {
            unsigned framediff = frame_counter - pb->frame_number_last_used;
            if (framediff != 0 && framediff != 1)
                continue;
        }

        if ((I & pb->getInterval()).count())
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
