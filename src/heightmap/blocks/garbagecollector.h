#ifndef HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H
#define HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H

#include "heightmap/blockcache.h"
#include "heightmap/glblock.h"

namespace Heightmap {
namespace Blocks {

class GarbageCollector
{
public:
    GarbageCollector(BlockCache::const_ptr cache);

    pBlock runOnce(unsigned frame_counter);
    std::vector<pBlock> runUntilComplete(unsigned frame_counter);
    std::vector<pBlock> releaseAllNotUsedInThisFrame(unsigned frame_counter);

private:
    BlockCache::const_ptr cache_;
};

} // namespace Block
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H
