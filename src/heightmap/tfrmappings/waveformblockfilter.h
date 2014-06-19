#ifndef HEIGHTMAP_TFRMAPPINGS_WAVEFORMBLOCKFILTER_H
#define HEIGHTMAP_TFRMAPPINGS_WAVEFORMBLOCKFILTER_H

#include "tfr/chunkfilter.h"
#include "heightmap/block.h"
#include "heightmap/update/mergechunk.h"
#include "heightmap/update/updatequeue.h"

namespace Heightmap {
namespace TfrMappings {

/**
 * @brief The WaveformBlockUpdater class should draw a waveform on a heightmap block.
 */
class WaveformBlockUpdater
{
public:
    class Job: public Update::IUpdateJob
    {
    public:
        Job(Signal::pMonoBuffer b) : b(b) {}

        Signal::pMonoBuffer b;

        Signal::Interval getCoveredInterval() const override { return b->getInterval (); }
    };


    void processJobs( const std::vector<Heightmap::Update::UpdateQueue::Job>& jobs );
    void processJob( const Job& job, const std::vector<pBlock>& intersecting_blocks );
    void processJob( const Job& job, pBlock block );
};


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
