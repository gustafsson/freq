#ifndef HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H

#include "mergechunk.h"
#include "tfr/chunkfilter.h"
#include "heightmap/block.h"

namespace Heightmap {
namespace TfrMappings {


class StftBlockFilterParams {
public:
    typedef shared_state<StftBlockFilterParams> ptr;

    Tfr::pChunkFilter freq_normalization;
};

/**
 * @brief The StftBlockFilter class should update a block with stft transform data.
 */
class StftBlockFilter: public Heightmap::MergeChunk
{
public:
    StftBlockFilter(StftBlockFilterParams::ptr params);

    std::vector<Update::IUpdateJob::ptr> prepareUpdate(Tfr::ChunkAndInverse&) override;

private:
    StftBlockFilterParams::ptr params_;

public:
    static void test();
};


/**
 * @brief The StftBlockFilterDesc class should instantiate StftBlockFilter for different engines.
 */
class StftBlockFilterDesc: public Heightmap::MergeChunkDesc
{
public:
    StftBlockFilterDesc(StftBlockFilterParams::ptr params);

    MergeChunk::ptr createMergeChunk(Signal::ComputingEngine* engine) const;

private:
    StftBlockFilterParams::ptr params_;

public:
    static void test();
};

} // namespace TfrMappings
} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H
