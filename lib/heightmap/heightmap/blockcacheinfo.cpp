#include "blockcacheinfo.h"

#include "render/glblock.h"
#include "tasktimer.h"

#include <boost/foreach.hpp>


namespace Heightmap {

unsigned long BlockCacheInfo::
        cacheByteSize(const BlockCache::cache_t& cache)
{
    // For each block there may be both a slope map and heightmap. Also there
    // may be both a texture and a vbo, and possibly a mapped cuda copy.
    //
    // But most of the blocks will only have a heightmap vbo, and none of the
    // others.

    unsigned long sumsize = 0;

    BOOST_FOREACH (const BlockCache::cache_t::value_type& b, cache)
        {
        if (b.second->glblock)
            sumsize += b.second->glblock->allocated_bytes_per_element()
                    * b.second->block_layout().texels_per_block ();
        }

    return sumsize;
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
