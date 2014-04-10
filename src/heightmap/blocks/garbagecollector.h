#ifndef HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H
#define HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H

#include "heightmap/blockcache.h"
#include "heightmap/glblock.h"

namespace Heightmap {
namespace Blocks {

class GarbageCollector
{
public:
    GarbageCollector(BlockCache::ptr cache);

    pGlBlock reuseOnOutOfMemory(unsigned frame_counter, const BlockLayout& block_layout);
    void releaseAllNotUsedInThisFrame(unsigned frame_counter);

private:
    BlockCache::ptr cache_;
};

} // namespace Block
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H
