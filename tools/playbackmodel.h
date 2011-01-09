#ifndef PLAYBACKMODEL_H
#define PLAYBACKMODEL_H

#include "signal/operation.h"
#include "signal/postsink.h"
#include "signal/worker.h"

#include <string>

namespace Adapters { class Playback; }

namespace Tools
{
    class SelectionModel;

    class PlaybackModel
    {
    public:
        PlaybackModel( SelectionModel* selection );

        Signal::PostSink* getPostSink();
        Signal::pWorkerCallback postsinkCallback;

        unsigned playback_device;
        std::string selection_filename;

        SelectionModel* selection;
        Signal::pOperation adapter_playback;

        Adapters::Playback* playback();
    };
} // namespace Tools
#endif // PLAYBACKMODEL_H
