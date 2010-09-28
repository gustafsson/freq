#include "playbackmodel.h"

#include "selectionmodel.h"

namespace Tools
{
    PlaybackModel::
            PlaybackModel( SelectionModel* selection )
                :
                playback_device(0),
                selection(selection),
                playback(0)
    {
    }

} // namespace Tools
