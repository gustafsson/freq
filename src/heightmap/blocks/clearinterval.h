#ifndef HEIGHTMAP_BLOCKS_CLEARINTERVAL_H
#define HEIGHTMAP_BLOCKS_CLEARINTERVAL_H

#include "heightmap/blockcache.h"

namespace Heightmap {
namespace Blocks {

class ClearInterval
{
public:
    ClearInterval(BlockCache::Ptr cache);

    /**
     * @brief discardOutside sets everything in cache to 0 outside the interval.
     * @param I the interval to keep.
     * @return A list of blocks that have been removed from the cache.
     */
    std::list<pBlock> discardOutside(Signal::Interval& I);

private:
    BlockCache::Ptr cache_;
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_CLEARINTERVAL_H
