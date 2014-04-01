#ifndef HEIGHTMAP_MERGECHUNK_H
#define HEIGHTMAP_MERGECHUNK_H

#include "tfr/chunkfilter.h"
#include "signal/computingengine.h"
#include "heightmap/ichunktoblock.h"
#include <vector>

namespace Heightmap {

class MergeChunk {
public:
    typedef shared_state<MergeChunk> Ptr;

    virtual ~MergeChunk() {}

    /**
     * @brief filterChunk is called from a worker thread.
     * May be empty.
     */
    virtual void filterChunk(Tfr::ChunkAndInverse&) {}

    /**
     * @brief createChunkToBlock will be called from the UI thread.
     */
    virtual std::vector<IChunkToBlock::Ptr> createChunkToBlock(Tfr::ChunkAndInverse&) = 0;
};


class MergeChunkDesc
{
public:
    typedef shared_state<MergeChunkDesc> Ptr;

    virtual ~MergeChunkDesc() {}

    virtual MergeChunk::Ptr createMergeChunk(Signal::ComputingEngine* engine=0) const = 0;
};

} // namespace Heightmap

#endif // HEIGHTMAP_MERGECHUNK_H
