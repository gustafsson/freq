#ifndef HEIGHTMAP_BLOCKS_CHUNKMERGER_H
#define HEIGHTMAP_BLOCKS_CHUNKMERGER_H

#include "heightmap/chunkblockfilter.h"
#include "tfr/chunkfilter.h"
#include "ichunkmerger.h"

#include <queue>

namespace Heightmap {
namespace Blocks {

/**
 * @brief The ChunkMerger class should merge chunks into blocks.
 */
class ChunkMerger: public IChunkMerger
{
public:
    ChunkMerger();

    // IChunkMerger
    void clear();
    void addChunk( MergeChunk::ptr merge_chunk,
                   Tfr::ChunkAndInverse chunk,
                   std::vector<pBlock> intersecting_blocks );
    bool processChunks(float timeout);

private:
    struct Job {
        MergeChunk::ptr merge_chunk;
        Tfr::ChunkAndInverse chunk;
        std::vector<pBlock> intersecting_blocks;
    };

    std::queue<Job> jobs;

    static void processJob(Job& j);

public:
    static void test();
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_CHUNKMERGER_H
