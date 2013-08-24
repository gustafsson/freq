#ifndef HEIGHTMAP_CHUNKTOBLOCK_H
#define HEIGHTMAP_CHUNKTOBLOCK_H

#include "block.h"
#include "tfr/filter.h"
#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"

namespace Heightmap {

class ChunkToBlock
{
public:
    ChunkToBlock();

    ComplexInfo complex_info;
    float normalization_factor;
    bool full_resolution;
    bool enable_subtexel_aggregation;
    Heightmap::TfrMapping tfr_mapping;

    void mergeColumnMajorChunk(
            const Block& block,
            Tfr::pChunk,
            BlockData& outData );

    void mergeRowMajorChunk(
            const Block& block,
            Tfr::pChunk,
            BlockData& outData );

    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKTOBLOCK_H
