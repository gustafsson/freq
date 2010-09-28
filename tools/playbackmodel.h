#ifndef PLAYBACKMODEL_H
#define PLAYBACKMODEL_H

#include <string>

namespace Signal { class Worker; }
namespace Adapters { class Playback; }

namespace Tools
{
    class SelectionModel;

    class PlaybackModel
    {
    public:
        PlaybackModel( SelectionModel* selection );

        unsigned playback_device;
        std::string selection_filename;

        SelectionModel* selection;
        Adapters::Playback* playback;
    };
} // namespace Tools
#endif // PLAYBACKMODEL_H
