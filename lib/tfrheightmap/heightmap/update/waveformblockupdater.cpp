#include "waveformblockupdater.h"
#include "tfr/drawnwaveformkernel.h"
#include "opengl/waveupdater.h"

using namespace std;

namespace Heightmap {
namespace Update {


class WaveformBlockUpdaterPrivate {
public:
    OpenGL::WaveUpdater w;
};

WaveformBlockUpdater::
        WaveformBlockUpdater()
    :
      p(new WaveformBlockUpdaterPrivate)
{

}


WaveformBlockUpdater::
        ~WaveformBlockUpdater()
{
    delete p;
}


void WaveformBlockUpdater::
        processJobs( std::queue<UpdateQueue::Job>& jobs )
{
    p->w.processJobs (jobs);
}


void WaveformBlockUpdater::
        processJobsCpu( std::queue<UpdateQueue::Job>& jobs )
{
    // Select subset to work on, must consume jobs in order
    std::vector<UpdateQueue::Job> myjobs;
    while (!jobs.empty ())
    {
        UpdateQueue::Job& j = jobs.front ();
        if (dynamic_cast<const WaveformBlockUpdater::Job*>(j.updatejob.get ()))
        {
            myjobs.push_back (std::move(j)); // Steal it
            jobs.pop ();
        }
        else
            break;
    }

    for (UpdateQueue::Job& job : myjobs)
      {
        auto bujob = dynamic_cast<const WaveformBlockUpdater::Job*>(job.updatejob.get ());
        processJob (*bujob, job.intersecting_blocks);
        job.promise.set_value ();
      }
}


void WaveformBlockUpdater::
        processJob( const WaveformBlockUpdater::Job& job,
                    const vector<pBlock>& intersecting_blocks )
{
    for (pBlock block : intersecting_blocks)
        processJob (job, block);
}


void WaveformBlockUpdater::
        processJob( const WaveformBlockUpdater::Job& job, pBlock block )
{
    Signal::pMonoBuffer b = job.b;
    float blobsize = b->sample_rate() / block->sample_rate();

    int readstop = b->number_of_samples ();

    // todo should substract blobsize/2
    Region r = block->getRegion ();
    float writeposoffs = (r.a.time - b->start ())*block->sample_rate ();
    float y0 = r.a.scale*2-1;
    float yscale = r.scale ()*2;
    auto d = block->block_data ();

    ::drawWaveform(
            b->waveform_data(),
            d->cpu_copy,
            blobsize,
            readstop,
            yscale,
            writeposoffs,
            y0);
}

} // namespace Update
} // namespace Heightmap
