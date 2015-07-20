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
    while (!jobs.empty ())
    {
        UpdateQueue::Job& j = jobs.front ();
        auto job = dynamic_cast<const WaveformBlockUpdater::Job*>(j.updatejob.get ());
        if (!job)
            break;

        auto f =
                [
                    wave2fbo = &p->wave2fbo,
                    b = job->b
                ]
                (const glProjection& M) mutable
                {
                    wave2fbo->draw (M,b);

                    return true;
                };

        for (pBlock block : j.intersecting_blocks)
        {
            block->updater ()->queueUpdate (block, f);

#ifdef PAINT_BLOCKS_FROM_UPDATE_THREAD
            block->updater ()->processUpdates (false);
#endif
        }

        INFO {
            auto job = dynamic_cast<const WaveformBlockUpdater::Job*>(j.updatejob.get ());
            Log("WaveUpdater finished %s") % job->b->getInterval();
        }

        j.promise.set_value ();
        jobs.pop ();
    }
}

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
