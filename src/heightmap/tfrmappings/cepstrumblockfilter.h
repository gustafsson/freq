#ifndef HEIGHTMAP_TFRMAPPINGS_CEPSTRUMBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_CEPSTRUMBLOCKFILTER_H

#include "tfr/chunkfilter.h"
#include "heightmap/block.h"
#include "heightmap/chunkblockfilter.h"

namespace Heightmap {
namespace TfrMappings {

class CepstrumBlockFilterParams: public VolatilePtr<CepstrumBlockFilterParams> {
public:
};

/**
 * @brief The StftBlockFilter class should update a block with stft transform data.
 */
class CepstrumBlockFilter: public Heightmap::MergeChunk
{
public:
    CepstrumBlockFilter(CepstrumBlockFilterParams::Ptr params);

    void filterChunk(Tfr::ChunkAndInverse&);
    std::vector<IChunkToBlock::Ptr> createChunkToBlock(Tfr::ChunkAndInverse&);

private:
    CepstrumBlockFilterParams::Ptr params_;

public:
    static void test();
};


/**
 * @brief The StftBlockFilterDesc class should instantiate StftBlockFilter for different engines.
 */
class CepstrumBlockFilterDesc: public Heightmap::MergeChunkDesc
{
public:
    CepstrumBlockFilterDesc(CepstrumBlockFilterParams::Ptr params = CepstrumBlockFilterParams::Ptr());

    MergeChunk::Ptr createMergeChunk(Signal::ComputingEngine* engine) const;

private:
    CepstrumBlockFilterParams::Ptr params_;

public:
    static void test();
};


} // namespace TfrMappings
} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPINGS_CEPSTRUMBLOCKFILTER_H
