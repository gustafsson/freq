#ifndef HEIGHTMAP_UPDATE_OPENGL_WAVEUPDATER_H
#define HEIGHTMAP_UPDATE_OPENGL_WAVEUPDATER_H

#include "heightmap/update/updatequeue.h"

namespace Heightmap {
namespace Update {
namespace OpenGL {

class WaveUpdaterPrivate;
class WaveUpdater
{
public:
    WaveUpdater();
    WaveUpdater(const WaveUpdater&) = delete;
    WaveUpdater& operator=(const WaveUpdater&) = delete;
    ~WaveUpdater();

    void processJobs( std::queue<UpdateQueue::Job>& jobs );
private:
    WaveUpdaterPrivate* p;
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_WAVEUPDATER_H
