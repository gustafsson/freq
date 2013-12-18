#ifndef HEIGHTMAP_TFRMAPPINGS_WAVEFORMBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_WAVEFORMBLOCKFILTER_H

#include "tfr/filter.h"
#include "heightmap/block.h"
#include "heightmap/chunkblockfilter.h"

namespace Heightmap {
namespace TfrMappings {

/**
 * @brief The WaveformBlockFilter class should draw a waveform on a heightmap block.
 */
class WaveformBlockFilter: public Heightmap::MergeChunk
{
private:
    void mergeChunk( const Heightmap::Block& block, const Tfr::ChunkAndInverse& chunk, Heightmap::BlockData& outData );

public:
    static void test();
};


/**
 * @brief The WaveformBlockFilterDesc class should instantiate WaveformBlockFilter for different engines.
 */
class WaveformBlockFilterDesc: public Heightmap::MergeChunkDesc
{
private:
    MergeChunk::Ptr createMergeChunk(Signal::ComputingEngine* engine) const;

public:
    static void test();
};

} // namespace TfrMappings
} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPINGS_WAVEFORMBLOCKFILTER_H
