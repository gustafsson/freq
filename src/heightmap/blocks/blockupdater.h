#ifndef HEIGHTMAP_BLOCKS_CHUNKMERGER_H
#define HEIGHTMAP_BLOCKS_CHUNKMERGER_H

#include "tfr/chunkfilter.h"
#include "updatequeue.h"

namespace Heightmap {
namespace Blocks {

/**
 * @brief The ChunkMerger class should update blocks with chunk data
 */
class BlockUpdater
{
public:
    BlockUpdater();

    void processJob( MergeChunk::ptr merge_chunk,
                     Tfr::ChunkAndInverse chunk,
                     std::vector<pBlock> intersecting_blocks );

public:
    static void test();
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_CHUNKMERGER_H
