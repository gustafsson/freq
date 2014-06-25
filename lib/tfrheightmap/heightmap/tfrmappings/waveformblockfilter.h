#ifndef HEIGHTMAP_TFRMAPPINGS_WAVEFORMBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_WAVEFORMBLOCKFILTER_H

#include "mergechunk.h"

namespace Heightmap {
namespace TfrMappings {

/**
 * @brief The WaveformBlockFilter class should prepare a waveform chunk for drawing.
 */
class WaveformBlockFilter: public Heightmap::MergeChunk
{
public:
    std::vector<Update::IUpdateJob::ptr> prepareUpdate(Tfr::ChunkAndInverse&);

public:
    static void test();
};


/**
 * @brief The WaveformBlockFilterDesc class should instantiate WaveformBlockFilter for different engines.
 */
class WaveformBlockFilterDesc: public Heightmap::MergeChunkDesc
{
private:
    MergeChunk::ptr createMergeChunk(Signal::ComputingEngine* engine) const;

public:
    static void test();
};

} // namespace TfrMappings
} // namespace Heightmap

#endif // HEIGHTMAP_TFRMAPPINGS_WAVEFORMBLOCKFILTER_H
