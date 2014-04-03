#ifndef HEIGHTMAP_TFRMAPPINGS_CWTBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_CWTBLOCKFILTER_H

#include "tfr/chunkfilter.h"
#include "heightmap/block.h"

namespace Heightmap {
namespace TfrMappings {

/**
 * @brief The CwtBlockFilter class should update a block with cwt transform data.
 */
class CwtBlockFilter: public Heightmap::MergeChunk
{
public:
    CwtBlockFilter(ComplexInfo complex_info);

    std::vector<IChunkToBlock::ptr> createChunkToBlock(Tfr::ChunkAndInverse&);

private:
    ComplexInfo complex_info_;

public:
    static void test();
};


/**
 * @brief The CwtBlockFilterDesc class should instantiate CwtBlockFilter for different engines.
 */
class CwtBlockFilterDesc: public Heightmap::MergeChunkDesc
{
public:
    CwtBlockFilterDesc(ComplexInfo complex_info);

    MergeChunk::ptr createMergeChunk(Signal::ComputingEngine* engine) const;

private:
    ComplexInfo complex_info_;

public:
    static void test();
};

} // namespace TfrMappings
} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPINGS_CWTBLOCKFILTER_H
