#include "waveupdater.h"
#include "heightmap/update/waveformblockupdater.h"
#include "heightmap/blockmanagement/blockupdater.h"
#include "heightmap/render/blocktextures.h"
#include "wave2fbo.h"
#include "lazy.h"
#include "log.h"

#include <unordered_map>

//#define INFO
#define INFO if(0)

using namespace std;
using namespace JustMisc;

namespace std {
    template<typename T>
    struct hash<boost::shared_ptr<T>>
    {
        size_t operator()(boost::shared_ptr<T> const& p) const
        {
            return hash<T*>()(p.get ());
        }
    };
}


namespace Heightmap {
namespace Update {
namespace OpenGL {

class WaveUpdaterPrivate
{
public:
    Wave2Fbo wave2fbo;
};


WaveUpdater::
        WaveUpdater()
    :
      p(new WaveUpdaterPrivate)
{
}


WaveUpdater::
        ~WaveUpdater()
{
    delete p;
}


void WaveUpdater::
        processJobs( queue<UpdateQueue::Job>& jobs )
{
    // Select subset to work on, must consume jobs in order
    vector<UpdateQueue::Job> myjobs;
    while (!jobs.empty ())
    {
        UpdateQueue::Job& j = jobs.front ();
        if (dynamic_cast<const WaveformBlockUpdater::Job*>(j.updatejob.get ()))
        {
            myjobs.push_back (move(j)); // Steal it
            jobs.pop ();
        }
        else
            break;
    }

    // Remap block -> buffer (instead of buffer -> blocks) because we want to draw all
    // buffers to each block, instead of each buffer to all blocks.
    //
    // The chunks must be drawn in order, thus a "vector<Tfr::pMonoBuffer>" is required
    // to preserve ordering.
    unordered_map<pBlock, vector<Signal::pMonoBuffer>> buffers_per_block;
    for (const UpdateQueue::Job& j : myjobs)
    {
        auto job = dynamic_cast<const WaveformBlockUpdater::Job*>(j.updatejob.get ());

        for (pBlock block : j.intersecting_blocks)
            buffers_per_block[block].push_back(job->b);
    }

    // Draw from all chunks to each block
    for (auto& f : buffers_per_block)
    {
        const pBlock& block = f.first;

        for (auto& b : f.second)
        {
            auto fc =
                [
                    wave2fbo = &p->wave2fbo,
                    b
                ]
                (const glProjection& M)
                {
                    wave2fbo->draw (M,b);
                    return true;
                };

            block->updater ()->queueUpdate (block, fc);
        }

        block->updater ()->processUpdates (false);
    }

    for (UpdateQueue::Job& j : myjobs) {
        INFO {
            auto job = dynamic_cast<const WaveformBlockUpdater::Job*>(j.updatejob.get ());
            Log("WaveUpdater finished %s") % job->b->getInterval();
        }

        j.promise.set_value ();
    }
}

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
