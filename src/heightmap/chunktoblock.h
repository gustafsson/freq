#ifndef HEIGHTMAP_CHUNKTOBLOCK_H
#define HEIGHTMAP_CHUNKTOBLOCK_H

#include "block.h"
#include "tfr/chunkfilter.h"
#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"
#include "ichunktoblock.h"

namespace Heightmap {

class ChunkToBlock: public IChunkToBlock
{
public:
    ChunkToBlock(Tfr::pChunk chunk);

    ComplexInfo complex_info;
    float normalization_factor;
    bool full_resolution;
    bool enable_subtexel_aggregation;

    void init() {}
    void prepareTransfer() {}
    void prepareMerge(AmplitudeAxis amplitude_axis, Tfr::FreqAxis display_scale, BlockLayout bl) {}
    void mergeChunk( pBlock block );

private:
    Tfr::pChunk chunk;

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

} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKTOBLOCK_H
