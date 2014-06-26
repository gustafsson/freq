#ifndef HEIGHTMAP_UPDATE_OPENGL_WAVEUPDATER_H
#define HEIGHTMAP_UPDATE_OPENGL_WAVEUPDATER_H

#include "heightmap/update/updatequeue.h"

namespace Heightmap {
namespace Update {
namespace OpenGL {

class WaveUpdater
{
public:
    WaveUpdater();
    WaveUpdater(const WaveUpdater&) = delete;
    WaveUpdater& operator=(const WaveUpdater&) = delete;
    ~WaveUpdater();

    void processJobs( std::queue<UpdateQueue::Job>& jobs );
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_WAVEUPDATER_H
