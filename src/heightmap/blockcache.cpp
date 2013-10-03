#include "blockcache.h"
#include "reference_hash.h"

#include "TaskTimer.h"

#include <boost/foreach.hpp>

namespace Heightmap {

BlockCache::
        BlockCache()
{
}


pBlock BlockCache::
        find( const Reference& ref )
{
    cache_t::const_iterator itr = cache_.find( ref );
    if (itr != cache_.end()) {
        pBlock b = itr->second;
        recent_.remove (b);
        recent_.push_front (b);
        return b;
    }

    // cache_misses_.insert(ref);

    return pBlock();
}


void BlockCache::
        insert( pBlock b )
{
    cache_[ b->reference() ] = b;
}


void BlockCache::
        erase (const Reference& ref)
{
    cache_t::iterator i = cache_.find(ref);
    if (i != cache_.end()) {
        recent_.remove(i->second);
        cache_.erase(i);
    }
}


void BlockCache::
        reset()
{
    cache_.clear();
    recent_.clear();
}


const BlockCache::cache_t& BlockCache::
        cache() const
{
    return cache_;
}


const BlockCache::recent_t& BlockCache::
        recent() const
{
    return recent_;
}


void BlockCache::
        test()
{
    // It should store allocated blocks readily available and register
    // asynchronous cache misses
    {
        Reference r1;
        Reference r2 = r1.right ();
        BlockLayout bl(2,2,1);
        VisualizationParams::Ptr vp;
        pBlock b1(new Block(r1, bl, vp));
        pBlock b2(new Block(r2, bl, vp));

        BlockCache c;
        c.insert (b1);
        c.insert (b2);
        pBlock b3 = c.find(r1);
        pBlock b4 = c.find(r2);

        EXCEPTION_ASSERT( b1 == b3 );
        EXCEPTION_ASSERT( b2 == b4 );
    }
}

} // namespace Heightmap
