#include "playbackcontroller.h"

#include "playbackview.h"
#include "playbackmodel.h"
#include "selectionmodel.h"
#include "renderview.h"

// Sonic AWE
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "adapters/writewav.h"
#include "adapters/playback.h"

namespace Tools
{

PlaybackController::
        PlaybackController( Sawe::Project* project, PlaybackView* view, RenderView* render_view )
            :
            _view(view)
{
    setupGui( project, render_view );
}


void PlaybackController::
        setupGui( Sawe::Project* project, RenderView* render_view )
{
    Ui::MainWindow* ui = project->mainWindow()->getItems();

    // User interface buttons
    connect(ui->actionPlaySelection, SIGNAL(triggered()), SLOT(receivePlaySound()));
    connect(ui->actionFollowPlayMarker, SIGNAL(triggered(bool)), SLOT(receiveFollowPlayMarker(bool)));

    // Make RenderView keep on rendering as long as the playback marker moves
    connect(_view, SIGNAL(update_view()), render_view, SLOT(update()));

    // If playback is active, draw the playback marker in PlaybackView whenever
    // RenderView paints.
    connect(render_view, SIGNAL(painting()), _view, SLOT(draw()));
    connect(render_view, SIGNAL(prePaint()), _view, SLOT(locatePlaybackMarker()));
}


void PlaybackController::
        receivePlaySound()
{
    TaskTimer tt("Initiating playback of selection");


    Signal::PostSink* selection_operations = _view->model->selection->getPostSink();

    // TODO define selections by a selection structure. Currently selections
    // are defined from the first sampels that is non-zero affected by a
    // filter, to the last non-zero affected sample.
    if (!selection_operations->filter()) {
        tt.info("No filter, no selection");
        return; // No filter, no selection...
    }

    if (selection_operations->sinks().empty())
    {
        std::vector<Signal::pOperation> sinks;
        sinks.push_back( Signal::pOperation( new Adapters::Playback( _view->model->playback_device )) );
        sinks.push_back( Signal::pOperation( new Adapters::WriteWav( _view->model->selection_filename )) );
        selection_operations->sinks( sinks );
    }

    _view->update();
}


void PlaybackController::
        receiveFollowPlayMarker( bool v )
{
    _view->follow_play_marker = v;
}


} // namespace Tools
