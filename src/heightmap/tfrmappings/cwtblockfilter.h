#ifndef HEIGHTMAP_TFRMAPPINGS_CWTBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_CWTBLOCKFILTER_H

#include "tfr/chunkfilter.h"
#include "heightmap/block.h"
#include "heightmap/chunkblockfilter.h"

namespace Heightmap {
namespace TfrMappings {

/**
 * @brief The CwtBlockFilter class should update a block with cwt transform data.
 */
class CwtBlockFilter: public Heightmap::MergeChunk
{
public:
    CwtBlockFilter(ComplexInfo complex_info);

private:
    void mergeChunk( const Heightmap::Block& block, const Tfr::ChunkAndInverse& chunk, Heightmap::BlockData& outData );

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

    MergeChunk::Ptr createMergeChunk(Signal::ComputingEngine* engine) const;

private:
    ComplexInfo complex_info_;

public:
    static void test();
};

} // namespace TfrMappings
} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPINGS_CWTBLOCKFILTER_H
