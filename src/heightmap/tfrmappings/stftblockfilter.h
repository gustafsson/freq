#ifndef HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H

#include "tfr/chunkfilter.h"
#include "heightmap/block.h"
#include "heightmap/chunkblockfilter.h"

namespace Heightmap {
namespace TfrMappings {


class StftBlockFilterParams {
public:
    typedef shared_state<StftBlockFilterParams> Ptr;

    Tfr::pChunkFilter freq_normalization;
};

/**
 * @brief The StftBlockFilter class should update a block with stft transform data.
 */
class StftBlockFilter: public Heightmap::MergeChunk
{
public:
    StftBlockFilter(StftBlockFilterParams::Ptr params);

    void filterChunk(Tfr::ChunkAndInverse&);
    std::vector<IChunkToBlock::Ptr> createChunkToBlock(Tfr::ChunkAndInverse&);

private:
    StftBlockFilterParams::Ptr params_;

public:
    static void test();
};


/**
 * @brief The StftBlockFilterDesc class should instantiate StftBlockFilter for different engines.
 */
class StftBlockFilterDesc: public Heightmap::MergeChunkDesc
{
public:
    StftBlockFilterDesc(StftBlockFilterParams::Ptr params);

    MergeChunk::Ptr createMergeChunk(Signal::ComputingEngine* engine) const;

private:
    StftBlockFilterParams::Ptr params_;

public:
    static void test();
};

} // namespace TfrMappings
} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H
