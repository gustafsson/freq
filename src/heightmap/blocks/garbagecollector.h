#ifndef HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H
#define HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H

#include "heightmap/blockcache.h"

namespace Heightmap {
namespace Blocks {

class GarbageCollector
{
public:
    GarbageCollector(BlockCache::Ptr cache);

    pBlock releaseOneBlock(unsigned frame_counter);
    void releaseAllNotUsedInThisFrame(unsigned _frame_counter);

private:
    BlockCache::Ptr cache_;
};

} // namespace Block
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H
