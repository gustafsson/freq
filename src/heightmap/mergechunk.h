#ifndef HEIGHTMAP_MERGECHUNK_H
#define HEIGHTMAP_MERGECHUNK_H

#include "tfr/chunkfilter.h"
#include "signal/computingengine.h"
#include "heightmap/ichunktoblock.h"
#include "shared_state_traits_backtrace.h"
#include <vector>

namespace Heightmap {

class MergeChunk {
public:
    typedef shared_state<MergeChunk> ptr;
    typedef shared_state_traits_backtrace shared_state_traits;

    virtual ~MergeChunk() {}

    /**
     * @brief filterChunk is called from a worker thread.
     * May be empty.
     */
    virtual void filterChunk(Tfr::ChunkAndInverse&) {}

    /**
     * @brief createChunkToBlock will be called from the UI thread.
     */
    virtual std::vector<IChunkToBlock::ptr> createChunkToBlock(Tfr::ChunkAndInverse&) = 0;
};


class MergeChunkDesc
{
public:
    typedef shared_state<MergeChunkDesc> ptr;

    virtual ~MergeChunkDesc() {}

    virtual MergeChunk::ptr createMergeChunk(Signal::ComputingEngine* engine=0) const = 0;
};

} // namespace Heightmap

#endif // HEIGHTMAP_MERGECHUNK_H
