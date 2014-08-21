#include "playbackcontroller.h"

// Tools
#include "playbackview.h"
#include "playbackmodel.h"
#include "selectionmodel.h"
#include "tools/renderview.h"
#include "playbackmarkersmodel.h"
#include "support/operation-composite.h"
#include "tools/support/toolbar.h"

// Sonic AWE
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "adapters/writewav.h"
#include "adapters/playback.h"
#include "tfr/chunkfilter.h"

// gpumisc
#include "demangle.h"

#include <QSettings>

namespace Tools
{

PlaybackController::Actions::
        Actions(QObject *parent)
{
    actionFollowPlayMarker = new QAction(parent);
    actionFollowPlayMarker->setObjectName(QStringLiteral("actionFollowPlayMarker"));
    actionFollowPlayMarker->setCheckable(true);
    QIcon icon14;
    icon14.addFile(QStringLiteral(":/icons/icons/icon-lockplayback.png"), QSize(), QIcon::Normal, QIcon::Off);
    actionFollowPlayMarker->setIcon(icon14);
    actionFollowPlayMarker->setIconVisibleInMenu(true);

    actionPlay = new QAction(parent);
    actionPlay->setObjectName(QStringLiteral("actionPlayEntireSound"));
    actionPlay->setCheckable(true);
    QIcon icon37;
    icon37.addFile(QStringLiteral(":/icons/icons/icon-play.png"), QSize(), QIcon::Normal, QIcon::Off);
    actionPlay->setIcon(icon37);

    actionRecord = new QAction(parent);
    actionRecord->setObjectName(QStringLiteral("actionRecord"));
    actionRecord->setCheckable(true);
    actionRecord->setChecked(false);
    actionRecord->setEnabled(false);
    actionRecord->setVisible(false);
    QIcon icon13;
    icon13.addFile(QStringLiteral(":/icons/icons/icon-record.png"), QSize(), QIcon::Normal, QIcon::Off);
    actionRecord->setIcon(icon13);
    actionRecord->setIconVisibleInMenu(true);

    actionFollowPlayMarker->setText(QApplication::translate("MainWindow", "Follow play marker", 0));
    actionFollowPlayMarker->setToolTip(QApplication::translate("MainWindow", "Follow the play marker when playing", 0));
    actionPlay->setText(QApplication::translate("MainWindow", "Play", 0));
    actionPlay->setToolTip(QApplication::translate("MainWindow", "Play [Space]", 0));
    actionPlay->setShortcut(QApplication::translate("MainWindow", "Space", 0));
    actionRecord->setText(QApplication::translate("MainWindow", "Record", 0));
    actionRecord->setToolTip(QApplication::translate("MainWindow", "Toggle recording [R]", 0));
    actionRecord->setShortcut(QApplication::translate("MainWindow", "R", 0));
}


PlaybackController::
        PlaybackController( Sawe::Project* project, PlaybackView* view, RenderView* render_view )
            :
            _view(view),
            project_( project )
{
    setupGui( render_view );
    addPlaybackToolbar( project_->mainWindow(), project_->mainWindow()->getItems()->menuToolbars );
}


void PlaybackController::
        setupGui( RenderView* render_view )
{
    if (!ui_items_)
        ui_items_.reset (new Actions(this));

    // User interface buttons
    connect(ui_items_->actionPlay,              SIGNAL(toggled(bool)), SLOT(play(bool)));
    connect(ui_items_->actionFollowPlayMarker,  SIGNAL(toggled(bool)), SLOT(followPlayMarker(bool)));

    // Make RenderView keep on rendering (with interactive framerate) as long
    // as the playback marker moves
    connect(_view, SIGNAL(update_view()), render_view, SLOT(redraw()));
    connect(_view, SIGNAL(playback_stopped()), SLOT(stop()), Qt::QueuedConnection);

    // If playback is active, draw the playback marker in PlaybackView whenever
    // RenderView paints.
    connect(render_view, SIGNAL(painting()), _view, SLOT(draw()));
    connect(render_view, SIGNAL(prePaint()), _view, SLOT(locatePlaybackMarker()));
    connect(_view->model->selection, SIGNAL(selectionChanged()), SLOT(onSelectionChanged()));
}


void PlaybackController::
        addPlaybackToolbar( QMainWindow* parent, QMenu* menu )
{
    Tools::Support::ToolBar* toolBarPlay = new Tools::Support::ToolBar(parent);
    toolBarPlay->setWindowTitle(QApplication::translate("MainWindow", "toolBar", 0));
    toolBarPlay->setObjectName(QStringLiteral("toolBarPlay"));
    parent->addToolBar( Qt::TopToolBarArea, toolBarPlay );

    toolBarPlay->addAction(ui_items_->actionPlay);
    toolBarPlay->addAction(ui_items_->actionRecord);
    toolBarPlay->addSeparator();
    toolBarPlay->addAction(ui_items_->actionFollowPlayMarker);

    QAction *actionToggleTimeControlToolBox;
    actionToggleTimeControlToolBox = new QAction(parent);
    actionToggleTimeControlToolBox->setObjectName(QStringLiteral("actionToggleTimeControlToolBox"));
    actionToggleTimeControlToolBox->setCheckable(true);
    actionToggleTimeControlToolBox->setChecked(true);
    menu->addAction(actionToggleTimeControlToolBox);

    actionToggleTimeControlToolBox->setText(QApplication::translate("MainWindow", "&Playback", 0));
    actionToggleTimeControlToolBox->setToolTip(QApplication::translate("MainWindow", "Toggle the playback control toolbox", 0));
    connect(actionToggleTimeControlToolBox, SIGNAL(toggled(bool)), toolBarPlay, SLOT(setVisible(bool)));
    connect(toolBarPlay, SIGNAL(visibleChanged(bool)), actionToggleTimeControlToolBox, SLOT(setChecked(bool)));
}


QAction *PlaybackController::
        actionRecord()
{
    return ui_items_->actionRecord;
}


//class NoZeros: public Signal::DeprecatedOperation
//{
//public:
//    NoZeros() : Signal::DeprecatedOperation(Signal::pOperation()) {}
//    Signal::Intervals zeroed_samples() { return Signal::Intervals(); }
//};


void PlaybackController::
        play( bool active )
{
    if (!active)
    {
        pause(true);
        return;
    }

    if (model()->adapter_playback)
    {
        pause(false);
        return;
    }

    TaskTimer tt("Initiating playback");

    ui_items_->actionPlay->setChecked( true );

    // startPlayback will insert it in the system so that the source is properly set    
    Signal::OperationDesc::ptr filter = _view->model->selection->current_selection_copy(SelectionModel::SaveInside_TRUE);
    if (!filter) {
        // here we just need to create a filter that does the right thing to an arbitrary source
        // and responds properly to zeroed_samples(), that is; a dummy Operation that doesn't do anything
        // and responds with no samples to zeroed_samples().
        //filter = Signal::OperationDesc::Ptr( new NoZeros() );
    }

    startPlayback( filter );
}


void PlaybackController::
        startPlayback ( Signal::OperationDesc::ptr filterdesc )
{
    Signal::Intervals zeroed_samples = Signal::Intervals();

    _view->just_started = true;

    if (filterdesc)
        TaskInfo("Selection: %s", filterdesc.read ()->toString().toStdString().c_str());

//    Signal::PostSink* postsink_operations = _view->model->playbackTarget->post_sink();
//    if ( postsink_operations->sinks().empty() || postsink_operations->filter() != filter )
    if ( true )
    {
        int playback_device = QSettings().value("outputdevice", -1).toInt();

        model()->adapter_playback.reset();
        Signal::Sink::ptr playbacksink(new Adapters::Playback( playback_device ));
        model()->adapter_playback.reset(new Signal::SinkDesc(playbacksink));

        Signal::OperationDesc::ptr desc(model()->adapter_playback);
        model()->target_marker = project_->processing_chain ().write ()->addTarget(desc, project_->default_target ());

        if (filterdesc)
            project_->processing_chain ().write ()->addOperationAt(filterdesc, model()->target_marker);

        Signal::OperationDesc::Extent x = project_->processing_chain ().write ()->extent(model()->target_marker);
        Signal::Intervals expected_data = ~zeroed_samples & x.interval.get_value_or (Signal::Interval());
        TaskInfo(boost::format("expected_data = %s") % expected_data);

        Adapters::Playback* playback = dynamic_cast<Adapters::Playback*>(model()->playback().get ());
        playback->setExpectedSamples (expected_data.spannedInterval (), x.number_of_channels.get_value_or (1));

        model()->target_marker->target_needs ()->updateNeeds(
                    expected_data,
                    Signal::Interval::IntervalType_MIN,
                    Signal::Interval::IntervalType_MAX,
                    1 );

        if (!expected_data)
            stop();
    }
    else
    {
        Adapters::Playback* playback = dynamic_cast<Adapters::Playback*>(model()->playback().get ());
        playback->restart_playback();
    }

    _view->update();
}


void PlaybackController::
        pause( bool active )
{
    if (model()->playback()) {
        Adapters::Playback* playback = dynamic_cast<Adapters::Playback*>(model()->playback().get ());
        playback->pausePlayback( active );
    }

    _view->update();
}


void PlaybackController::
        followPlayMarker( bool v )
{
    _view->follow_play_marker = v;
    // doesn't need to call update as new frames are continously rendered
    // during playback anyway
}


void PlaybackController::
        onSelectionChanged()
{
    stop();

    model()->target_marker.reset();
    model()->adapter_playback.reset();
}


void PlaybackController::
        stop()
{
    //TaskInfo("PlaybackController::receiveStop()");

    Signal::Operation::ptr p = model()->playback();
    if (p) {
        Adapters::Playback* playback = dynamic_cast<Adapters::Playback*>(model()->playback().get ());
        playback->stop();
    }

    model()->target_marker.reset();
    model()->adapter_playback.reset();

    _view->just_started = false;
    ui_items_->actionPlay->setChecked( false );

    _view->update();
}


PlaybackModel* PlaybackController::
        model()
{
    return _view->model;
}


} // namespace Tools
