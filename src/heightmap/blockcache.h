#ifndef HEIGHTMAP_BLOCKCACHE_H
#define HEIGHTMAP_BLOCKCACHE_H

#include "block.h"

// gpumisc
#include "shared_state.h"

// boost
#include <boost/unordered_map.hpp>

namespace Heightmap {

/**
 * @brief The BlockCache class should store allocated blocks readily available
 */
class BlockCache
{
public:
    typedef shared_state<BlockCache> Ptr;
    typedef shared_state<const BlockCache> ConstPtr;

    typedef boost::unordered_map<Reference, pBlock> cache_t;
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
     * @brief probe tests if 'ref' can be found in the BlockCache.
     * @param ref the block to search for.
     * @return a block if it is found or pBlock() otherwise.
     */
    pBlock      probe( const Reference& ref ) const;


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
     * @brief clear empties this cache.
     */
    void        clear();

    const cache_t& cache() const;
    const recent_t& recent() const;

private:


    /**
      The cache contains as many blocks as there are space for in the GPU ram.
      If allocation of a new block fails to be allocated
            1) all unused blocks are freed.
            2) if no unused blocks are found and _cache is non-empty, the entire _cache is cleared.
            3) if _cache is empty, Sonic AWE is terminated with an OpenGL or Cuda error.
      */

    cache_t         cache_;
    recent_t        recent_;     /// Ordered with the most recently accessed blocks first

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKCACHE_H
