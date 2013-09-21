#ifndef HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H

#include "tfr/filter.h"
#include "heightmap/block.h"
#include "heightmap/chunkblockfilter.h"

namespace Heightmap {
namespace TfrMappings {

/**
 * @brief The StftBlockFilter class should update a block with stft transform data.
 */
class StftBlockFilter: public Heightmap::MergeChunk
{
    void mergeChunk( const Heightmap::Block& block, const Tfr::ChunkAndInverse& chunk, Heightmap::BlockData& outData );

public:
    static void test();
};


/**
 * @brief The StftBlockFilterDesc class should instantiate StftBlockFilter for different engines.
 */
class StftBlockFilterDesc: public Heightmap::MergeChunkDesc
{
    MergeChunk::Ptr createMergeChunk(Signal::ComputingEngine* engine) const;

public:
    static void test();
};

} // namespace TfrMappings
} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPINGS_STFTBLOCKFILTER_H
