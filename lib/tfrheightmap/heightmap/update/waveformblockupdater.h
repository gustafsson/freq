#ifndef HEIGHTMAP_UPDATE_WAVEFORMBLOCKUPDATER_H
#define HEIGHTMAP_UPDATE_WAVEFORMBLOCKUPDATER_H

#include "updatequeue.h"
#include "signal/buffer.h"

namespace Heightmap {
namespace Update {

/**
 * @brief The WaveformBlockUpdater class should draw a waveform on a heightmap block.
 */
class WaveformBlockUpdaterPrivate;
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

    WaveformBlockUpdater();
    WaveformBlockUpdater(WaveformBlockUpdater const&)=delete;
    WaveformBlockUpdater& operator=(WaveformBlockUpdater const&)=delete;
    ~WaveformBlockUpdater();

    void processJobs( std::queue<Heightmap::Update::UpdateQueue::Job>& jobs );
    void processJobsCpu( std::queue<Heightmap::Update::UpdateQueue::Job>& jobs );
    void processJob( const Job& job, const std::vector<pBlock>& intersecting_blocks );
    void processJob( const Job& job, pBlock block );
private:
    WaveformBlockUpdaterPrivate* p;
};

} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_WAVEFORMBLOCKUPDATER_H
