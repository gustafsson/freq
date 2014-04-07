#ifndef HEIGHTMAP_TFRMAPPINGS_CEPSTRUMBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_CEPSTRUMBLOCKFILTER_H

#include "tfr/chunkfilter.h"
#include "heightmap/block.h"
#include "heightmap/mergechunk.h"

namespace Heightmap {
namespace TfrMappings {

class CepstrumBlockFilterParams {
public:
    typedef shared_state<CepstrumBlockFilterParams> ptr;
};

/**
 * @brief The StftBlockFilter class should update a block with stft transform data.
 */
class CepstrumBlockFilter: public Heightmap::MergeChunk
{
public:
    CepstrumBlockFilter(CepstrumBlockFilterParams::ptr params);

    std::vector<Blocks::IUpdateJob::ptr> prepareUpdate(Tfr::ChunkAndInverse&) override;

private:
    CepstrumBlockFilterParams::ptr params_;

public:
    static void test();
};


/**
 * @brief The StftBlockFilterDesc class should instantiate StftBlockFilter for different engines.
 */
class CepstrumBlockFilterDesc: public Heightmap::MergeChunkDesc
{
public:
    CepstrumBlockFilterDesc(CepstrumBlockFilterParams::ptr params = CepstrumBlockFilterParams::ptr());

    MergeChunk::ptr createMergeChunk(Signal::ComputingEngine* engine) const;

private:
    CepstrumBlockFilterParams::ptr params_;

public:
    static void test();
};


} // namespace TfrMappings
} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPINGS_CEPSTRUMBLOCKFILTER_H
