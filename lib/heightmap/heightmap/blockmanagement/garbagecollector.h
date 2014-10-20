#ifndef HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H
#define HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H

#include "heightmap/blockcache.h"
#include <functional>
#include <set>

namespace Heightmap {
namespace Blocks {

class GarbageCollector
{
public:
    GarbageCollector(BlockCache::const_ptr cache);

    unsigned countBlocksUsedThisFrame(unsigned frame_counter);
    pBlock getOldestBlock(unsigned frame_counter);
    std::vector<pBlock> getNOldest(unsigned frame_counter, unsigned N);

    /**
     * @brief getAllNotUsedInThisFrame finds unused blocks
     * @param frame_counter
     * @return all blocks whose 'frame_number_last_used' doesn't match frame_counter.
     */
    std::vector<pBlock> getAllNotUsedInThisFrame(unsigned frame_counter);

private:
    BlockCache::const_ptr cache_;

    typedef std::set<pBlock, std::function<bool(const pBlock&, const pBlock&)>> Sorted;

    Sorted getSorted(unsigned frame_counter);
};

} // namespace Block
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCK_GARBAGECOLLECTOR_H
