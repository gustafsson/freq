#include "playbackmodel.h"

#include "selectionmodel.h"

#include "adapters/playback.h"
#include "sawe/project.h"

namespace Tools
{
    PlaybackModel::
            PlaybackModel(Sawe::Project* /*project*/)
                :
//                playbackTarget(new Signal::Target(&project->layers, "Playback", false, true)),
                selection(0),
                markers(0)
    {
    }


    Signal::Operation::ptr PlaybackModel::
            playback()
    {
        if (!adapter_playback)
            return Signal::Operation::ptr();

        auto s = adapter_playback.read ();
        const Signal::SinkDesc* sinkdesc = dynamic_cast<const Signal::SinkDesc*>(&*s);
        return sinkdesc->sink ();
    }

} // namespace Tools
