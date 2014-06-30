#ifndef HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H
#define HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H

#include "heightmap/blockcache.h"
#include "heightmap/render/glblock.h"
#include <functional>
#include <set>

namespace Heightmap {
namespace Blocks {

class GarbageCollector
{
public:
    GarbageCollector(BlockCache::const_ptr cache);

    unsigned countBlocksUsedThisFrame(unsigned frame_counter);
    pBlock runOnce(unsigned frame_counter);
    std::vector<pBlock> runUntilComplete(unsigned frame_counter);
    std::vector<pBlock> releaseNOldest(unsigned frame_counter, unsigned N);
    std::vector<pBlock> releaseAllNotUsedInThisFrame(unsigned frame_counter);

private:
    BlockCache::const_ptr cache_;

    typedef std::set<pBlock, std::function<bool(const pBlock&, const pBlock&)>> Sorted;

    Sorted getSorted(unsigned frame_counter);
};

} // namespace Block
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H
