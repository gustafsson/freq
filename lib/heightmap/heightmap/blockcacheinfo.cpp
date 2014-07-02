#include "blockcacheinfo.h"

#include "render/blocktextures.h"
#include "tasktimer.h"


namespace Heightmap {

unsigned long BlockCacheInfo::
        cacheByteSize(const BlockCache::cache_t& cache)
{
    if (cache.empty ())
        return 0;

    pBlock b = cache.begin ()->second;
    unsigned bytes_per_block =
            Render::BlockTextures::allocated_bytes_per_element() *
            b->block_layout().texels_per_block ();

    return cache.size () * bytes_per_block;
}


void BlockCacheInfo::
        printCacheSize(const BlockCache::cache_t& cache)
{
    size_t free=0, total=0;
#ifdef USE_CUDA
    cudaMemGetInfo(&free, &total);
#endif

    TaskInfo("Currently has %u cached blocks (ca %s). There are %s graphics memory free of a total of %s",
             cache.size(),
             DataStorageVoid::getMemorySizeText( cacheByteSize(cache) ).c_str(),
             DataStorageVoid::getMemorySizeText( free ).c_str(),
             DataStorageVoid::getMemorySizeText( total ).c_str());
}


void BlockCacheInfo::
        test()
{

}

} // namespace Heightmap
