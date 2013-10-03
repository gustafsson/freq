#ifndef HEIGHTMAP_BLOCKCACHE_H
#define HEIGHTMAP_BLOCKCACHE_H

#include "block.h"

// gpumisc
#include "volatileptr.h"

// boost
#include <boost/unordered_map.hpp>

namespace Heightmap {

/**
 * @brief The BlockCache class should store allocated blocks readily available
 * and register asynchronous cache misses.
 */
class BlockCache: public VolatilePtr<BlockCache>
{
public:
    typedef boost::unordered_map<Reference, pBlock> cache_t;
    //typedef std::set<Reference> cache_misses_t;
    typedef std::list<pBlock> recent_t;

    BlockCache();

    /**
     * @brief findBlock searches through the cache for a block with reference 'ref'
     * @param ref the block to search for.
     * @return a block if it is found or pBlock() otherwise.
     *
     * Is not const because it updates a list of recently accessed blocks.
     */
    pBlock      find( const Reference& ref );

    /**
     * @brief insertBlock
     * @param b
     */
    void        insert( pBlock b );

    /**
     * @brief removeBlock
     * @param b
     */
    void        erase( const Reference& ref );

    /**
     * @brief reset empties this cache.
     */
    void        reset();

    const cache_t& cache() const;
    const recent_t& recent() const;
    //const recent_t& to_remove() const { return to_remove_; }
    //const cache_misses_t& cache_misses() { return cache_misses_; }
    //void clear_cache_misses() { cache_misses_.clear(); }


private:


    /**
      The cache contains as many blocks as there are space for in the GPU ram.
      If allocation of a new block fails to be allocated
            1) all unused blocks are freed.
            2) if no unused blocks are found and _cache is non-empty, the entire _cache is cleared.
            3) if _cache is empty, Sonic AWE is terminated with an OpenGL or Cuda error.
      */

    cache_t         cache_;
    //cache_misses_t  cache_misses_;
    recent_t        recent_;     /// Ordered with the most recently accessed blocks first

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKCACHE_H
