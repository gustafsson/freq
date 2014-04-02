#ifndef PLAYBACKCONTROLLER_H
#define PLAYBACKCONTROLLER_H

#include "signal/operation.h"

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
        void play( bool active );
        void pause( bool active );
        void stop();
        void followPlayMarker( bool v );
        void onSelectionChanged();

    private:
        PlaybackModel* model();
        PlaybackView* _view;

        Sawe::Project* project_;

        struct Actions {
            Actions(QObject* parent);

            QAction *actionFollowPlayMarker;
            QAction *actionPausePlayBack;
            QAction *actionPlay;
            QAction *actionRecord;
            QAction *actionStop;
        };
        boost::shared_ptr<Actions> ui_items_;

        void startPlayback ( Signal::OperationDesc::ptr filter );

        // GUI
        void setupGui( RenderView* render_view );
        void addPlaybackToolbar( QMainWindow* parent, QMenu* menu );
    };
} // namespace Tools
#endif // PLAYBACKCONTROLLER_H
