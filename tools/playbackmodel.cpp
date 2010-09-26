#include "playbackmodel.h"

namespace Tools
{
    PlaybackModel::
            PlaybackModel( Signal::Worker* worker )
                :
                playback_device(0),
                selection(0),
                worker(worker),
                playback(0)
    {
    }

} // namespace Tools
