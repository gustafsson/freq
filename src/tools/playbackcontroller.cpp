#include "playbackcontroller.h"

// Tools
#include "playbackview.h"
#include "playbackmodel.h"
#include "selectionmodel.h"
#include "renderview.h"
#include "playbackmarkersmodel.h"
#include "support/operation-composite.h"
#include "signal/oldoperationwrapper.h"

// Sonic AWE
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "adapters/writewav.h"
#include "adapters/playback.h"
#include "tfr/filter.h"

// gpumisc
#include "demangle.h"

#include <QSettings>

namespace Tools
{

PlaybackController::
        PlaybackController( Sawe::Project* project, PlaybackView* view, RenderView* render_view )
            :
            _view(view),
            project_( project ),
            ui_items_( project_->mainWindow()->getItems() )
{
    setupGui( render_view );
}


void PlaybackController::
        setupGui( RenderView* render_view )
{
    connect(ui_items_->actionToggleTimeControlToolBox, SIGNAL(toggled(bool)), ui_items_->toolBarPlay, SLOT(setVisible(bool)));
    connect(ui_items_->toolBarPlay, SIGNAL(visibleChanged(bool)), ui_items_->actionToggleTimeControlToolBox, SLOT(setChecked(bool)));

    // User interface buttons
    connect(ui_items_->actionPlaySelection, SIGNAL(toggled(bool)), SLOT(receivePlaySelection(bool)));
    connect(ui_items_->actionPlaySection, SIGNAL(toggled(bool)), SLOT(receivePlaySection(bool)));
    connect(ui_items_->actionPlayEntireSound, SIGNAL(toggled(bool)), SLOT(receivePlayEntireSound(bool)));

    connect(ui_items_->actionPausePlayBack, SIGNAL(toggled(bool)), SLOT(receivePause(bool)));
    connect(ui_items_->actionStopPlayBack, SIGNAL(triggered()), SLOT(receiveStop()));
    connect(ui_items_->actionFollowPlayMarker, SIGNAL(toggled(bool)), SLOT(receiveFollowPlayMarker(bool)));

    // Make RenderView keep on rendering (with interactive framerate) as long
    // as the playback marker moves
    connect(_view, SIGNAL(update_view(bool)), render_view, SLOT(userinput_update(bool)));
    connect(_view, SIGNAL(playback_stopped()), SLOT(receiveStop()));

    // If playback is active, draw the playback marker in PlaybackView whenever
    // RenderView paints.
    connect(render_view, SIGNAL(painting()), _view, SLOT(draw()));
    connect(render_view, SIGNAL(prePaint()), _view, SLOT(locatePlaybackMarker()));
    connect(render_view, SIGNAL(populateTodoList()), SLOT(populateTodoList()));
    connect(_view->model->selection, SIGNAL(selectionChanged()), SLOT(onSelectionChanged()));
}


void PlaybackController::
        receivePlaySelection( bool active )
{
    if (!active)
    {
        receiveStop();
        return;
    }

    TaskTimer tt("Initiating playback of selection");

    ui_items_->actionPlaySection->setChecked( false );
    ui_items_->actionPlayEntireSound->setChecked( false );
    ui_items_->actionPausePlayBack->setChecked( false );
    ui_items_->actionPlaySelection->setChecked( true );

    Signal::pOperation filter = _view->model->selection->current_selection_copy(SelectionModel::SaveInside_TRUE);

    startPlayback( filter );
}


void PlaybackController::
        receivePlaySection( bool /*active*/ )
{
    EXCEPTION_ASSERTX(false, "not implemented");
#if 0
    if (!active)
    {
        receiveStop();
        return;
    }

    TaskTimer tt("Initiating playback of section");

    ui_items_->actionPlaySelection->setChecked( false );
    ui_items_->actionPlayEntireSound->setChecked( false );
    ui_items_->actionPausePlayBack->setChecked( false );
    ui_items_->actionPlaySection->setChecked( true );

    // startPlayback will insert it in the system so that the source is properly set
    // here we just need to create a filter that does the right thing to an arbitrary source
    // and responds properly to zeroed_samples()
    Signal::pOperation filter( new Support::OperationOtherSilent(
            project_->head->head_source(),
            _view->model->markers->currentInterval( project_->head->head_source()->sample_rate() ) ));

    startPlayback( filter );
#endif
}


class NoZeros: public Signal::DeprecatedOperation
{
public:
    NoZeros() : Signal::DeprecatedOperation(Signal::pOperation()) {}
    Signal::Intervals zeroed_samples() { return Signal::Intervals(); }
};


void PlaybackController::
        receivePlayEntireSound( bool active )
{
    if (!active)
    {
        receiveStop();
        return;
    }

    TaskTimer tt("Initiating playback");

    ui_items_->actionPlaySelection->setChecked( false );
    ui_items_->actionPlaySection->setChecked( false );
    ui_items_->actionPausePlayBack->setChecked( false );
    ui_items_->actionPlayEntireSound->setChecked( true );

    // startPlayback will insert it in the system so that the source is properly set    
    Signal::pOperation filter = _view->model->selection->current_selection_copy(SelectionModel::SaveInside_TRUE);
    if (!filter) {
        // here we just need to create a filter that does the right thing to an arbitrary source
        // and responds properly to zeroed_samples(), that is; a dummy Operation that doesn't do anything
        // and responds with no samples to zeroed_samples().
        filter = Signal::pOperation( new NoZeros() );
    }

    startPlayback( filter );
}


class ExtentOp : public Signal::DeprecatedOperation
{
public:
    ExtentOp(Signal::OperationDesc::Extent x)
        :
          Signal::DeprecatedOperation(Signal::pOperation()),
          x(x)
    {}

    virtual Signal::IntervalType number_of_samples() {
        return x.interval.get_value_or (Signal::Interval()).last;
    }

    virtual unsigned num_channels() {
        return x.number_of_channels.get_value_or (1);
    }

    virtual float sample_rate() {
        return x.sample_rate.get_value_or (1);
    }

private:
    Signal::OperationDesc::Extent x;
};


void PlaybackController::
        startPlayback ( Signal::pOperation filter )
{
    if (!filter) {
        TaskInfo("No filter, no selection");
        receiveStop();
        return; // No filter, no selection...
    }

    _view->just_started = true;
    ui_items_->actionPausePlayBack->setEnabled( true );

    TaskInfo("Selection is of type %s", filter->toStringSkipSource ().c_str());

//    Signal::PostSink* postsink_operations = _view->model->playbackTarget->post_sink();
//    if ( postsink_operations->sinks().empty() || postsink_operations->filter() != filter )
    if ( true )
    {
        int playback_device = QSettings().value("outputdevice", -1).toInt();

        model()->adapter_playback.reset();
        model()->adapter_playback.reset( new Adapters::Playback( playback_device ));

        Signal::OperationDesc::Ptr desc(new Signal::OldOperationDescWrapper(model()->adapter_playback) );
        model()->target_marker = write1(project_->processing_chain ())->addTarget(desc, project_->default_target ());

//        std::vector<Signal::pOperation> sinks;
//        postsink_operations->sinks( sinks ); // empty
//        sinks.push_back( model()->adapter_playback );
        //sinks.push_back( Signal::pOperation( new Adapters::WriteWav( _view->model->selection_filename )) );

//        postsink_operations->filter( Signal::pOperation() );
//        postsink_operations->sinks( sinks );
//        postsink_operations->filter( filter );

        //Signal::Intervals expected_data = ~filter->zeroed_samples_recursive();
        Signal::OperationDesc::Extent x = write1(project_->processing_chain ())->extent(model()->target_marker);
        Signal::pOperation extent_op(new ExtentOp(x));
        filter->source (extent_op);
        Signal::Intervals expected_data = ~filter->zeroed_samples() & x.interval.get_value_or (Signal::Interval());
        TaskInfo(boost::format("expected_data = %s") % expected_data);
        TaskInfo(boost::format("filter->zeroed_samples() = %s") % filter->zeroed_samples());
        Signal::OperationDesc::Ptr filterdesc(new Signal::OldOperationDescWrapper(filter) );
        write1(project_->processing_chain ())->addOperationAt(filterdesc, model()->target_marker);

        model()->playback ()->setExpectedSamples (expected_data.spannedInterval (), x.number_of_channels.get_value_or (1));
        write1(model()->target_marker->target_needs ())->updateNeeds(
                    expected_data,
                    Signal::Interval::IntervalType_MIN,
                    Signal::Interval::IntervalType_MAX,
                    Signal::Intervals(),
                    1 );

        if (!expected_data)
            receiveStop();
    }
    else
    {
        model()->playback()->restart_playback();
    }

    _view->update();
}


void PlaybackController::
        receivePause( bool active )
{
    if (active)
    {
        if (!ui_items_->actionPlaySelection->isChecked() &&
            !ui_items_->actionPlaySection->isChecked() &&
            !ui_items_->actionPlayEntireSound->isChecked())
        {
            ui_items_->actionPausePlayBack->setChecked( false );
        }
    }

    model()->playback()->pausePlayback( active );
}


void PlaybackController::
        receiveFollowPlayMarker( bool v )
{
    _view->follow_play_marker = v;
    // doesn't need to call update as new frames are continously rendered
    // during playback anyway
}


void PlaybackController::
        onSelectionChanged()
{
    if (ui_items_->actionPlaySelection->isChecked())
        receiveStop();

    ui_items_->actionPlaySelection->setEnabled( 0 != _view->model->selection->current_selection() );

//    std::vector<Signal::pOperation> empty;
//    model()->playbackTarget->post_sink()->sinks( empty );
//    model()->playbackTarget->post_sink()->filter( Signal::pOperation() );
    model()->target_marker.reset();
    model()->adapter_playback.reset();
}


void PlaybackController::
        populateTodoList()
{
    EXCEPTION_ASSERTX(false, "Use Signal::Processing namespace");
/*
    Signal::Intervals missing_for_playback=
            model()->playbackTarget->post_sink()->invalid_samples();

    if (missing_for_playback)
    {
        bool playback_is_underfed = project_->tools().playback_model.playbackTarget->post_sink()->isUnderfed();

        // Don't bother with computing playback unless it is underfed
        if (playback_is_underfed)
        {
            project_->worker.center = 0;
            project_->worker.target( project_->tools().playback_model.playbackTarget );
        }
    }
*/
}


void PlaybackController::
        receiveStop()
{
    TaskInfo("PlaybackController::receiveStop()");

    if (model()->playback())
        model()->playback()->stop();

    model()->target_marker.reset();
    model()->adapter_playback.reset();

    _view->just_started = false;
    ui_items_->actionPlaySelection->setChecked( false );
    ui_items_->actionPlaySection->setChecked( false );
    ui_items_->actionPlayEntireSound->setChecked( false );
    ui_items_->actionPausePlayBack->setChecked( false );
    ui_items_->actionPausePlayBack->setEnabled( false );

    _view->update();
}


PlaybackModel* PlaybackController::
        model()
{
    return _view->model;
}


} // namespace Tools
