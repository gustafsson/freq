#include "playbackmodel.h"

#include "selectionmodel.h"

#include "adapters/playback.h"
#include "sawe/project.h"

namespace Tools
{
    PlaybackModel::
            PlaybackModel(Sawe::Project* project)
                :
                playbackTarget(new Signal::Target(&project->layers, "Playback")),
                playback_device(-1),
                selection(0),
                markers(0)
    {
    }


    Adapters::Playback* PlaybackModel::
            playback()
    {
        return dynamic_cast<Adapters::Playback*>(adapter_playback.get());
    }

} // namespace Tools
