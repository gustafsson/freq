#ifndef PLAYBACKCONTROLLER_H
#define PLAYBACKCONTROLLER_H

#include "signal/operation.h"

#include <QObject>

namespace Sawe { class Project; }
namespace Ui { class MainWindow; }

namespace Tools
{
    class PlaybackModel;
    class PlaybackView;
    class RenderView;

    class PlaybackController: public QObject
    {
        Q_OBJECT
    public:
        PlaybackController( Sawe::Project* project, PlaybackView* view, RenderView* render_view );

    private slots:
        void receivePlaySelection( bool active );
        void receivePlaySection( bool active );
        void receivePlayEntireSound( bool active );
        void receivePause( bool active );
        void receiveStop();
        void receiveFollowPlayMarker( bool v );
        void onSelectionChanged();

    private:
        PlaybackModel* model();
        PlaybackView* _view;

        Sawe::Project* project_;
        Ui::MainWindow* ui_items_;

        void startPlayback ( Signal::pOperation filter );

        // GUI
        void setupGui( RenderView* render_view );

    };
} // namespace Tools
#endif // PLAYBACKCONTROLLER_H
