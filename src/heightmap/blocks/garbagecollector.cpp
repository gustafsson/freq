#include "garbagecollector.h"
#include "heightmap/blockcacheinfo.h"
#include "heightmap/referenceinfo.h"
#include "heightmap/glblock.h"
#include "heightmap/visualizationparams.h"

#include "tasktimer.h"
#include "computationkernel.h"

// Limit the amount of memory used for caches by memoryUsedForCaches < freeMemory*MAX_FRACTION_FOR_CACHES
#define MAX_FRACTION_FOR_CACHES (1.f/2.f)

using namespace boost;

namespace Heightmap {
namespace Blocks {

GarbageCollector::
        GarbageCollector(BlockCache::Ptr cache)
    :
      cache_(cache)
{
}


pBlock GarbageCollector::
        releaseOneBlock(unsigned _frame_counter)
{
    size_t _free_memory = availableMemoryForSingleAllocation();

    auto cache = cache_.write ();
    // prefer to use block rather than discard an old block and then reallocate it

    // _recent is ordered with the most recently accessed blocks first,
    pBlock block;
    if (!cache->recent ().empty())
    {
        pBlock stealedBlock = cache->recent ().back();

        // blocks that were used last frame has age 1 and blocks that were used this frame has age 0
        unsigned age = _frame_counter - stealedBlock->frame_number_last_used;

        if (1 < age)
        {
            size_t memForNewBlock = 0;
            memForNewBlock += sizeof(float); // OpenGL VBO
            memForNewBlock += sizeof(float); // Cuda device memory
            memForNewBlock += sizeof(float); // OpenGL texture
            memForNewBlock += sizeof(float); // OpenGL neareset texture
            memForNewBlock *= stealedBlock->block_layout ().texels_per_block ();
            size_t allocatedMemory = BlockCacheInfo::cacheByteSize (cache->cache ());

            size_t margin = 2*memForNewBlock;

            if (allocatedMemory+memForNewBlock+margin > _free_memory*MAX_FRACTION_FOR_CACHES)
            {
                pBlock stealedBlock = cache->recent ().back();

                TaskInfo ti(format("Stealing block %s last used %u frames ago. Total free %s, total cache %s, %u blocks")
                            % stealedBlock->reference ()
                            % (_frame_counter - stealedBlock->frame_number_last_used)
                            % DataStorageVoid::getMemorySizeText( _free_memory )
                            % DataStorageVoid::getMemorySizeText( allocatedMemory )
                            % cache->cache ().size()
                             );

                block = stealedBlock;

                cache->erase (stealedBlock->reference ());
            }

            // Need to release even more blocks? Release one at a time for each call to createBlock
            if (allocatedMemory > _free_memory*MAX_FRACTION_FOR_CACHES)
            {
                pBlock back = cache->recent ().back();

                EXCEPTION_ASSERT_LESS_OR_EQUAL(3, back.use_count ());

                if (back.use_count () == 3) // recent, cache and back
                {
                    size_t blockMemory = back->glblock->allocated_bytes_per_element()*block->block_layout ().texels_per_block ();
                    allocatedMemory -= std::min(allocatedMemory,blockMemory);
                    _free_memory = _free_memory > blockMemory ? _free_memory + blockMemory : 0;

                    TaskInfo(format("Removing block %s last used %u frames ago. Freeing %s, total free %s, cache %s, %u blocks")
                                 % back->reference ()
                                 % (_frame_counter - back->frame_number_last_used)
                                 % DataStorageVoid::getMemorySizeText( blockMemory )
                                 % DataStorageVoid::getMemorySizeText( _free_memory )
                                 % DataStorageVoid::getMemorySizeText( allocatedMemory )
                                 % cache->cache ().size()
                                 );

                    cache->erase (back->reference ());
                }
            }
        }
    }

    return block;
}


void GarbageCollector::
        releaseAllNotUsedInThisFrame(unsigned _frame_counter)
{
    auto cache = cache_.write ();
    const BlockCache::cache_t C = cache->cache (); // copy
    TaskTimer tt("Collection doing garbage collection", C.size());
    BlockCacheInfo::printCacheSize(C);

    for (BlockCache::cache_t::const_iterator itr = C.begin(); itr!=C.end(); itr++ )
    {
        EXCEPTION_ASSERT_LESS_OR_EQUAL(4, itr->second.use_count ());
        if (itr->second.use_count () == 4) // recent, cache, C and itr->second
        {
            if (_frame_counter != itr->second->frame_number_last_used )
                cache->erase( itr->first );
        }
    }
}

} // namespace Block
} // namespace Heightmap
