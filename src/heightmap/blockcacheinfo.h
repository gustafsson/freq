#ifndef HEIGHTMAP_BLOCKCACHEINFO_H
#define HEIGHTMAP_BLOCKCACHEINFO_H

#include "blockcache.h"

namespace Heightmap {

class BlockCacheInfo
{
public:
    static unsigned long cacheByteSize(const BlockCache::cache_t& cache);
    static void printCacheSize(const BlockCache::cache_t& cache);

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKCACHEINFO_H
