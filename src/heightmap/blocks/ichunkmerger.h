#ifndef HEIGHTMAP_BLOCKS_ICHUNKMERGER_H
#define HEIGHTMAP_BLOCKS_ICHUNKMERGER_H

#include "volatileptr.h"
#include "heightmap/mergechunk.h"
#include "tfr/chunkfilter.h"

namespace Heightmap {
namespace Blocks {

class IChunkMerger: public VolatilePtr<IChunkMerger>
{
public:
    virtual ~IChunkMerger() {}

    virtual void clear() = 0;
    virtual void addChunk( MergeChunk::Ptr merge_chunk,
                   Tfr::ChunkAndInverse chunk,
                   std::vector<pBlock> intersecting_blocks ) = 0;

    /**
     * @brief processChunks
     * @param timeout
     * @return true if finished within timeout.
     */
    virtual bool processChunks(float timeout) volatile = 0;
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_ICHUNKMERGER_H