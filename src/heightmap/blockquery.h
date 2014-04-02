#ifndef HEIGHTMAP_BLOCKQUERY_H
#define HEIGHTMAP_BLOCKQUERY_H

#include "blockcache.h"

namespace Heightmap {

/**
 * @brief The BlockQuery class should answer questions about block the cache.
 */
class BlockQuery
{
public:
    BlockQuery(BlockCache::const_ptr cache);

    /**
     * @brief getIntersectingBlocks

      Blocks are updated by CwtToBlock and StftToBlock by merging chunks into
      all existing blocks that intersect with the chunk interval.

      This method is called by working threads.

     * @param I
     * @param only_visible
     * @param frame_counter doesn't matter if only_visible is false
     * @return
     */
    std::vector<pBlock> getIntersectingBlocks( const Signal::Intervals& I, bool only_visible, int frame_counter ) const;

private:
    BlockCache::const_ptr cache_;
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKQUERY_H
