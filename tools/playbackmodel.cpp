#include "playbackmodel.h"

#include "selectionmodel.h"

#include "adapters/playback.h"

namespace Tools
{
    PlaybackModel::
            PlaybackModel( SelectionModel* selection )
                :
                playback_device(0),
                selection(selection)
    {
    }


    Adapters::Playback* PlaybackModel::
            playback()
    {
        return dynamic_cast<Adapters::Playback*>(adapter_playback.get());
    }

} // namespace Tools
