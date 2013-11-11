#ifndef PLAYBACKCONTROLLER_H
#define PLAYBACKCONTROLLER_H

#include "signal/poperation.h"

#include <QObject>

class QAction;
class QMainWindow;
class QMenu;

namespace Sawe { class Project; }
namespace Ui { class MainWindow; }

namespace Tools
{
    namespace Support { class ToolBar; }

    class PlaybackModel;
    class PlaybackView;
    class RenderView;

    class PlaybackController: public QObject
    {
        Q_OBJECT
    public:
        PlaybackController( Sawe::Project* project, PlaybackView* view, RenderView* render_view );

        QAction *actionRecord();

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

        struct Actions {
            Actions(QObject* parent);

            QAction *actionFollowPlayMarker;
            QAction *actionPausePlayBack;
            QAction *actionPlayEntireSound;
            QAction *actionPlaySection;
            QAction *actionPlaySelection;
            QAction *actionRecord;
            QAction *actionSetPlayMarker;
            QAction *actionStopPlayBack;
        };
        boost::shared_ptr<Actions> ui_items_;

        void startPlayback ( Signal::pOperation filter );

        // GUI
        void setupGui( RenderView* render_view );
        void addPlaybackToolbar( QMainWindow* parent, QMenu* menu );
    };
} // namespace Tools
#endif // PLAYBACKCONTROLLER_H
