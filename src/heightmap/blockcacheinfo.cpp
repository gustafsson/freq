#include "blockcacheinfo.h"

#include "glblock.h"
#include "TaskTimer.h"

#include <boost/foreach.hpp>


namespace Heightmap {

unsigned long BlockCacheInfo::
        cacheByteSize(const BlockCache::cache_t& cache, const BlockLayout& block_layout)
{
    // For each block there may be both a slope map and heightmap. Also there
    // may be both a texture and a vbo, and possibly a mapped cuda copy.
    //
    // But most of the blocks will only have a heightmap vbo, and none of the
    // others.

    unsigned long sumsize = 0;

    BOOST_FOREACH (const BlockCache::cache_t::value_type& b, cache)
        {
        sumsize += b.second->glblock->allocated_bytes_per_element();
        }

//    BOOST_FOREACH (const pBlock& b, to_remove)
//        {
//        unsigned abpe = b->glblock->allocated_bytes_per_element();
//        if (abpe > sumsize)
//            sumsize = 0;
//        else
//            sumsize -= abpe;
//        }

    unsigned elements_per_block = block_layout.texels_per_block ();
    return sumsize*elements_per_block;
}


void BlockCacheInfo::
        printCacheSize(const BlockCache::cache_t& cache, const BlockLayout& block_layout)
{
    size_t free=0, total=0;
#ifdef USE_CUDA
    cudaMemGetInfo(&free, &total);
#endif

    TaskInfo("Currently has %u cached blocks (ca %s). There are %s graphics memory free of a total of %s",
             cache.size(),
             DataStorageVoid::getMemorySizeText( cacheByteSize(cache, block_layout) ).c_str(),
             DataStorageVoid::getMemorySizeText( free ).c_str(),
             DataStorageVoid::getMemorySizeText( total ).c_str());
}


void BlockCacheInfo::
        test()
{

}

} // namespace Heightmap
