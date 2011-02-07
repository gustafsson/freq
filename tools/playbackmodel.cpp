#include "playbackmodel.h"

#include "selectionmodel.h"

#include "adapters/playback.h"
#include "sawe/project.h"

namespace Tools
{
    PlaybackModel::
            PlaybackModel(Sawe::Project* project)
                :
                playback_device(-1),
                selection(0),
                markers(0)
    {
        postsinkCallback.reset( new Signal::WorkerCallback(
                &project->worker,
                Signal::pOperation(new Signal::PostSink)) );
    }


    Signal::PostSink* PlaybackModel::
            getPostSink()
    {
        return dynamic_cast<Signal::PostSink*>(postsinkCallback->sink().get());
    }


    Adapters::Playback* PlaybackModel::
            playback()
    {
        return dynamic_cast<Adapters::Playback*>(adapter_playback.get());
    }

} // namespace Tools
