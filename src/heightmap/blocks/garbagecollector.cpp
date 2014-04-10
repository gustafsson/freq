#include "garbagecollector.h"
#include "heightmap/blockcacheinfo.h"
#include "heightmap/referenceinfo.h"
#include "heightmap/glblock.h"
#include "heightmap/visualizationparams.h"
#include "heightmap/reference_hash.h"

#include "tasktimer.h"
#include "computationkernel.h"

// Limit the amount of memory used for caches by memoryUsedForCaches < freeMemory*MAX_FRACTION_FOR_CACHES
#define MAX_FRACTION_FOR_CACHES (1.f/2.f)

using namespace boost;

namespace Heightmap {
namespace Blocks {

GarbageCollector::
        GarbageCollector(BlockCache::ptr cache)
    :
      cache_(cache)
{
}


pBlock getOldestBlock(unsigned frame_counter, BlockCache::cache_t& cache, unsigned min_age) {
    auto i = std::max_element(cache.begin(), cache.end(),
    [frame_counter](const BlockCache::cache_t::value_type& a, const BlockCache::cache_t::value_type& b) {
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


pGlBlock GarbageCollector::
        reuseOnOutOfMemory(unsigned frame_counter, const BlockLayout& block_layout)
{
    size_t memForNewBlock = 0;
    memForNewBlock += sizeof(float); // OpenGL VBO
    memForNewBlock += sizeof(float); // Cuda device memory
    memForNewBlock += sizeof(float); // OpenGL texture
    memForNewBlock += sizeof(float); // OpenGL neareset texture
    memForNewBlock *= block_layout.texels_per_block ();
    size_t free_memory = availableMemoryForSingleAllocation();
    BlockCache::cache_t cacheCopy = cache_->clone();
    size_t allocatedMemory = BlockCacheInfo::cacheByteSize (cacheCopy);

    size_t margin = 2*memForNewBlock;

    if (allocatedMemory+memForNewBlock+margin < free_memory*MAX_FRACTION_FOR_CACHES)
        return pGlBlock(); // No need to reuse

    pBlock stealedBlock = getOldestBlock(frame_counter, cacheCopy, 2);
    if (!stealedBlock)
        return pGlBlock(); // Nothing to reuse

    cache_->erase(stealedBlock->reference ());

    if (allocatedMemory > free_memory*MAX_FRACTION_FOR_CACHES)
    {
        cacheCopy.erase(stealedBlock->reference ());

        if (pBlock releasedBlock = getOldestBlock(frame_counter, cacheCopy, 2))
        {
            Heightmap::pGlBlock glblock = releasedBlock->glblock;
            size_t blockMemory = glblock ? glblock->allocated_bytes_per_element()*block_layout.texels_per_block () : 0;

            TaskInfo(format("Removing block %s last used %u frames ago. Freeing %s, total free %s, cache %s, %u blocks")
                         % stealedBlock->reference ()
                         % (frame_counter - stealedBlock->frame_number_last_used)
                         % DataStorageVoid::getMemorySizeText( blockMemory )
                         % DataStorageVoid::getMemorySizeText( free_memory )
                         % DataStorageVoid::getMemorySizeText( allocatedMemory )
                         % cacheCopy.size()
                         );

            releasedBlock->glblock.reset ();
            cache_->erase(releasedBlock->reference ());
        }
    }

    TaskInfo ti(format("Stealing block %s last used %u frames ago. Total free %s, total cache %s, %u blocks")
                % stealedBlock->reference ()
                % (frame_counter - stealedBlock->frame_number_last_used)
                % DataStorageVoid::getMemorySizeText( free_memory )
                % DataStorageVoid::getMemorySizeText( allocatedMemory )
                % cacheCopy.size()
                 );

    auto g = stealedBlock->glblock;
    stealedBlock->glblock.reset ();
    return g;
}


void GarbageCollector::
        releaseAllNotUsedInThisFrame(unsigned frame_counter)
{
    const BlockCache::cache_t C = cache_->clone (); // copy
    TaskTimer tt("Collection doing garbage collection", C.size());
    BlockCacheInfo::printCacheSize(C);

    for (BlockCache::cache_t::const_iterator itr = C.begin(); itr!=C.end(); itr++ )
    {
        EXCEPTION_ASSERT_LESS_OR_EQUAL(4, itr->second.use_count ());
        if (itr->second.use_count () == 4) // recent, cache, C and itr->second
        {
            if (frame_counter != itr->second->frame_number_last_used )
                cache_->erase( itr->first );
        }
    }
}

} // namespace Block
} // namespace Heightmap
