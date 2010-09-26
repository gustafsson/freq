#ifndef PLAYBACKCONTROLLER_H
#define PLAYBACKCONTROLLER_H

#include <QObject>

namespace Sawe { class Project; }

namespace Tools
{
    class PlaybackModel;
    class PlaybackView;
    class RenderView;

    class PlaybackController: QObject
    {
        Q_OBJECT
    public:
        PlaybackController( Sawe::Project* project, PlaybackView* view, RenderView* render_view );

    private slots:
        virtual void receivePlaySound();
        virtual void receiveFollowPlayMarker( bool v );

    private:
        // View (and model inside view)
        PlaybackView* _view;

        // GUI
        void setupGui( Sawe::Project* project, RenderView* render_view );

    };
} // namespace Tools
#endif // PLAYBACKCONTROLLER_H
