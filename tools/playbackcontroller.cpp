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
#include "tfr/filter.h"

// gpumisc
#include "demangle.h"

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

    // Make RenderView keep on rendering (with interactive framerate) as long
    // as the playback marker moves
    connect(_view, SIGNAL(update_view()), render_view, SLOT(userinput_update()));

    // If playback is active, draw the playback marker in PlaybackView whenever
    // RenderView paints.
    connect(render_view, SIGNAL(painting()), _view, SLOT(draw()));
    connect(render_view, SIGNAL(prePaint()), _view, SLOT(locatePlaybackMarker()));
}


void PlaybackController::
        receivePlaySound()
{
    TaskTimer tt("Initiating playback of selection");


    Signal::PostSink* postsink_operations = _view->model->getPostSink();
    Signal::pOperation filter = _view->model->selection->current_selection();

    // TODO define selections by a selection structure. Currently selections
    // are defined from the first sampels that is non-zero affected by a
    // filter, to the last non-zero affected sample.
    if (!filter) {
        tt.info("No filter, no selection");
        return; // No filter, no selection...
    }


    if ( postsink_operations->sinks().empty() || postsink_operations->filter() != filter )
    {
        model()->adapter_playback.reset();
        model()->adapter_playback.reset( new Adapters::Playback( _view->model->playback_device ));

        std::vector<Signal::pOperation> sinks;
        postsink_operations->sinks( sinks );
        sinks.push_back( model()->adapter_playback );
        sinks.push_back( Signal::pOperation( new Adapters::WriteWav( _view->model->selection_filename )) );

        postsink_operations->filter( Signal::pOperation() );
        postsink_operations->sinks( sinks );
        postsink_operations->filter( filter );
        postsink_operations->invalidate_samples( ~filter->zeroed_samples() );
    }
    else
    {
        model()->playback()->restart_playback();
    }

    _view->update();
}


void PlaybackController::
        receiveFollowPlayMarker( bool v )
{
    _view->follow_play_marker = v;
    // don't need to call update as new frames are continously rendered
    // during playback anyway
}


PlaybackModel* PlaybackController::
        model()
{
    return _view->model;
}


} // namespace Tools
