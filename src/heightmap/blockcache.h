#ifndef HEIGHTMAP_BLOCKCACHE_H
#define HEIGHTMAP_BLOCKCACHE_H

#include "block.h"
#include "reference_hash.h"

#include <unordered_map>
#include <thread>

namespace Heightmap {

/**
 * @brief The BlockCache class should store allocated blocks readily available
 */
class BlockCache
{
public:
    typedef std::shared_ptr<BlockCache> ptr;
    typedef std::shared_ptr<const BlockCache> const_ptr;
    typedef std::unordered_map<Reference, pBlock> cache_t;
    typedef std::list<pBlock> recent_t;


    BlockCache();

    /**
     * @brief findBlock searches through the cache for a block with reference 'ref'
     * @param ref the block to search for.
     * @return a block if it is found or pBlock() otherwise.
     */
    pBlock      find( const Reference& ref ) const;


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
     * @brief clear empties this cache and returns the contents it had. The return value can safely be discarded.
     */
    cache_t     clear();

    size_t      size() const;

    bool        empty() const;

    cache_t     clone() const;

private:


    /**
      The cache contains as many blocks as there are space for in the GPU ram.
      If allocation of a new block fails to be allocated
            1) all unused blocks are freed.
            2) if no unused blocks are found and _cache is non-empty, the entire _cache is cleared.
            3) if _cache is empty, Sonic AWE is terminated with an OpenGL or Cuda error.
      */

    mutable std::mutex  mutex_;
    cache_t             cache_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKCACHE_H
