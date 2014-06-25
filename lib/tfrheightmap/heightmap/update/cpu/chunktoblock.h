#ifndef HEIGHTMAP_CHUNKTOBLOCK_H
#define HEIGHTMAP_CHUNKTOBLOCK_H

#include "heightmap/block.h"
#include "tfr/chunkfilter.h"
#include "tfr/freqaxis.h"
#include "heightmap/amplitudeaxis.h"

namespace Heightmap {
namespace Update {

class ChunkToBlock
{
public:
    ChunkToBlock();

    ComplexInfo complex_info;
    float normalization_factor;
    bool full_resolution;
    bool enable_subtexel_aggregation;

    void mergeChunk( Tfr::pChunk chunk, pBlock block );

private:
    void mergeColumnMajorChunk(
            const Block& block,
            const Tfr::Chunk&,
            BlockData& outData );

    void mergeRowMajorChunk(
            const Block& block,
            const Tfr::Chunk&,
            BlockData& outData );

public:
    static void test();
};

} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKTOBLOCK_H
