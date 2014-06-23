#ifndef HEIGHTMAP_MERGECHUNK_H
#define HEIGHTMAP_MERGECHUNK_H

#include "signal/computingengine.h"
#include "heightmap/update/iupdatejob.h"
#include "tfr/chunkfilter.h"

#include "shared_state.h"

#include <vector>

namespace Heightmap {

class MergeChunk {
public:
    typedef std::shared_ptr<MergeChunk> ptr;

    virtual ~MergeChunk() {}

    virtual std::vector<Update::IUpdateJob::ptr> prepareUpdate(Tfr::ChunkAndInverse&) = 0;
    virtual std::vector<Update::IUpdateJob::ptr> prepareUpdate(Tfr::ChunkAndInverse& cai, const std::vector<pBlock>&) {
        return prepareUpdate (cai);
    }
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
