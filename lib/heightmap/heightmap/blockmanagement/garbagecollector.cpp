#include "garbagecollector.h"
#include "heightmap/blockcacheinfo.h"
#include "heightmap/referenceinfo.h"
#include "heightmap/render/glblock.h"
#include "heightmap/visualizationparams.h"
#include "heightmap/reference_hash.h"
#include "heightmap/render/blocktextures.h"

#include "tasktimer.h"
#include "log.h"
#include "computationkernel.h"
#include "neat_math.h"

#include <functional>

// Limit the amount of memory used for caches by memoryUsedForCaches < freeMemory*MAX_FRACTION_FOR_CACHES
#define MAX_FRACTION_FOR_CACHES (1.f/8.f)

#define INFO
//#define INFO if(0)

using namespace boost;

namespace Heightmap {
namespace Blocks {


pBlock getOldestBlock(unsigned frame_counter, BlockCache::cache_t& cache, unsigned min_age) {
    typedef const BlockCache::cache_t::value_type pair;
    auto i = std::max_element(cache.begin(), cache.end(), [frame_counter](pair& a, pair& b) {
            unsigned age_a = frame_counter - a.second->frame_number_last_used;
            unsigned age_b = frame_counter - b.second->frame_number_last_used;
            return age_a < age_b;
        });

    if (i == cache.end())
        return pBlock();

    unsigned age = frame_counter - i->second->frame_number_last_used;
    if (age < min_age)
        return pBlock();

    return i->second;
}


GarbageCollector::
        GarbageCollector(BlockCache::const_ptr cache)
    :
      cache_(cache)
{
}


unsigned GarbageCollector::
        countBlocksUsedThisFrame(unsigned frame_counter)
{
    const BlockCache::cache_t C = cache_->clone (); // copy

    unsigned i = 0;
    for (const BlockCache::cache_t::value_type& v : C)
        i += frame_counter == v.second->frame_number_last_used;

    return i;
}


pBlock GarbageCollector::
        runOnce(unsigned frame_counter)
{
    size_t free_memory = availableMemoryForSingleAllocation();
    BlockCache::cache_t cacheCopy = cache_->clone();
    size_t allocatedMemory = BlockCacheInfo::cacheByteSize (cacheCopy);

    if (allocatedMemory < free_memory*MAX_FRACTION_FOR_CACHES)
        return pBlock(); // No need to release memory

    pBlock releasedBlock = getOldestBlock(frame_counter, cacheCopy, 2);
    if (!releasedBlock)
        return pBlock(); // Nothing to release

    size_t blockMemory = Render::BlockTextures::allocated_bytes_per_element() * releasedBlock->block_layout().texels_per_block ();

    if (true)
    TaskInfo(format("Removing block %s last used %u frames ago. Freeing %s, total free %s, cache %s, %u blocks")
                 % releasedBlock->getRegion ()
                 % (frame_counter - releasedBlock->frame_number_last_used)
                 % DataStorageVoid::getMemorySizeText( blockMemory )
                 % DataStorageVoid::getMemorySizeText( free_memory )
                 % DataStorageVoid::getMemorySizeText( allocatedMemory )
                 % cacheCopy.size()
                 );

    return releasedBlock;
}


std::vector<pBlock> GarbageCollector::
        runUntilComplete(unsigned frame_counter)
{
    BlockCache::cache_t cacheCopy = cache_->clone();
    size_t free_memory = availableMemoryForSingleAllocation();
    size_t allocatedMemory = BlockCacheInfo::cacheByteSize (cacheCopy);

    auto sorted = getSorted(frame_counter);

    std::vector<pBlock> R;
    R.reserve (sorted.size ());

    // Go from oldest to newest
    for (pBlock b : sorted)
    {
        if (allocatedMemory < free_memory*MAX_FRACTION_FOR_CACHES)
            break;

        size_t blockMemory = Render::BlockTextures::allocated_bytes_per_element() * b->block_layout().texels_per_block ();
        allocatedMemory = clamped_sub(allocatedMemory, blockMemory);
        R.push_back (b);
    }

    return R;
}


std::vector<pBlock> GarbageCollector::
        releaseNOldest(unsigned frame_counter, unsigned N)
{
    auto sorted = getSorted(frame_counter);

    std::vector<pBlock> R;
    R.reserve (N);

    // Go from oldest to newest
    for (pBlock b : sorted)
        if (R.size () < N)
            R.push_back (b);

    return R;
}


std::vector<pBlock> GarbageCollector::
        releaseAllNotUsedInThisFrame(unsigned frame_counter)
{
    std::vector<pBlock> R;
    const BlockCache::cache_t C = cache_->clone (); // copy
    TaskTimer tt("Collection doing garbage collection. Cache size %u", C.size());
    BlockCacheInfo::printCacheSize(C);

    for (BlockCache::cache_t::const_iterator itr = C.begin(); itr!=C.end(); itr++ )
    {
        EXCEPTION_ASSERT_LESS_OR_EQUAL(4, itr->second.use_count ());
        if (itr->second.use_count () == 4) // recent, cache, C and itr->second
        {
            if (frame_counter != itr->second->frame_number_last_used )
                R.push_back (itr->second);
        }
    }

    return R;
}


GarbageCollector::Sorted GarbageCollector::
        getSorted(unsigned frame_counter)
{
    size_t free_memory = availableMemoryForSingleAllocation();
    BlockCache::cache_t cacheCopy = cache_->clone();
    size_t allocatedMemory = BlockCacheInfo::cacheByteSize (cacheCopy);

    // Sort with decreasing age
    GarbageCollector::Sorted sorted (
            [frame_counter](const pBlock& a, const pBlock& b) {
                unsigned age_a = frame_counter - a->frame_number_last_used;
                unsigned age_b = frame_counter - b->frame_number_last_used;
                if (age_a == age_b)
                    return a < b;
                return age_a > age_b;
            }
    );

    if (allocatedMemory < free_memory*MAX_FRACTION_FOR_CACHES)
        return sorted; // No need to release memory

    for (auto& v : cacheCopy) {
        unsigned age = frame_counter - v.second->frame_number_last_used;
        if (age>1) // Initial filtering
            sorted.insert (v.second);
    }

    return sorted;
}


} // namespace Block
} // namespace Heightmap
