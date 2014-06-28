#include "waveformblockupdater.h"
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

} // namespace Update
} // namespace Heightmap
