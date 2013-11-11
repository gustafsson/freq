#ifndef HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H

#include "tfr/filter.h"
#include "heightmap/block.h"
#include "heightmap/chunkblockfilter.h"

namespace Heightmap {
namespace TfrMappings {


class StftBlockFilterParams: public VolatilePtr<StftBlockFilterParams> {
public:
    Tfr::pChunkFilter freq_normalization;
};

/**
 * @brief The StftBlockFilter class should update a block with stft transform data.
 */
class StftBlockFilter: public Heightmap::MergeChunk
{
public:
    StftBlockFilter(StftBlockFilterParams::Ptr params);

    void prepareChunk(Tfr::ChunkAndInverse& chunk);

    void mergeChunk( const Heightmap::Block& block, const Tfr::ChunkAndInverse& chunk, Heightmap::BlockData& outData );

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
