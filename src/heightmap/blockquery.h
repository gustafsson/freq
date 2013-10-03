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
    BlockQuery(BlockCache::Ptr cache);

    std::vector<pBlock> getIntersectingBlocks( const Signal::Intervals& I, bool only_visible, int frame_counter ) const;

private:
    BlockCache::Ptr cache_;
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKQUERY_H
