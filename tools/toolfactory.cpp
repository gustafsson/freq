#include "toolfactory.h"

// Tools
#include "navigationcontroller.h"
#include "rendercontroller.h"
#include "renderview.h"
#include "selectioncontroller.h"
#include "selectionview.h"
#include "timelinecontroller.h"
#include "timelineview.h"
#include "playbackcontroller.h"
#include "playbackview.h"
#include "brushcontroller.h"
#include "brushview.h"
#include "recordmodel.h"
#include "recordcontroller.h"
#include "recordview.h"
#include "commentcontroller.h"
#include "matlabcontroller.h"
#include "graphcontroller.h"
#include "tooltipcontroller.h"
#include "aboutdialog.h"
#include "playbackmarkerscontroller.h"
#include "transforminfoform.h"
#include "exportaudiodialog.h"
#include "harmonicsinfoform.h"
#include "workercontroller.h"
#include "fantrackercontroller.h"
#include "fantrackerview.h"

// Sonic AWE
#include "sawe/project.h"
#include "ui/mainwindow.h"

// gpumisc
#include <TaskTimer.h>

// Qt
#include <QHBoxLayout>

namespace Tools
{

ToolFactory::
        ToolFactory(Sawe::Project* p)
:   ToolRepo(p),
    render_model( p ),
    selection_model( p ),
    playback_model( p )
{
    _render_view = new RenderView(&render_model);
    _render_controller.reset( new RenderController(_render_view) );

    _timeline_view = new TimelineView(p, _render_view);
    _timeline_controller = new TimelineController(_timeline_view);

    _selection_controller = new SelectionController(&selection_model, _render_view );

    _navigation_controller = new NavigationController(_render_view);

    playback_model.selection = &selection_model;
    _playback_view.reset( new PlaybackView(&playback_model, _render_view) );
    _playback_controller = new PlaybackController(p, _playback_view.data(), _render_view);

#ifndef TARGET_sss
    // No brushes for Sjostridsskolan, the Swedish Naval Academy
        _brush_model.reset( new BrushModel(p, &render_model) );
        _brush_view.reset( new BrushView(_brush_model.data() ));
        _brush_controller = new BrushController( _brush_view.data(), _render_view );
#endif

    if (RecordModel::canCreateRecordModel(p))
    {
        _record_model.reset( new RecordModel(p, _render_view) );
        _record_view.reset( new RecordView(_record_model.data() ));
        _record_controller = new RecordController( _record_view.data() );
    }

    _comment_controller = new CommentController( _render_view );
    tool_controllers_.push_back( _comment_controller );

    _matlab_controller = new MatlabController( p, _render_view );

    _graph_controller = new GraphController( _render_view );

    _tooltip_controller = new TooltipController(
            _render_view, dynamic_cast<CommentController*>(_comment_controller.data()) );
    tool_controllers_.push_back( _tooltip_controller );

    _fantracker_view.reset(new FanTrackerView());
    _fantracker_controller = new FanTrackerController(_fantracker_view.data(), _render_view );

    _about_dialog = new AboutDialog( p );

    _playbackmarkers_model.reset( new PlaybackMarkersModel() );
    _playbackmarkers_view.reset( new PlaybackMarkersView( _playbackmarkers_model.data(), p ));
    _playbackmarkers_controller = new PlaybackMarkersController(
            _playbackmarkers_view.data(), _render_view );
    playback_model.markers = _playbackmarkers_model.data();

    _transform_info_form = new TransformInfoForm(p, _render_controller.data() );

    _export_audio_dialog = new ExportAudioDialog(p, &selection_model, _render_view);

    _harmonics_info_form = new HarmonicsInfoForm(
            p,
            dynamic_cast<TooltipController*>(_tooltip_controller.data()),
            _render_view
            );

    _worker_view.reset( new WorkerView(p));
    _worker_controller.reset( new WorkerController( _worker_view.data(), _render_view, _timeline_view ) );
}


ToolFactory::
        ~ToolFactory()
{
    TaskInfo ti(__FUNCTION__);
    // Try to clear things in the opposite order that they were created

    // 'delete 0' is a valid operation and does nothing

    delete _harmonics_info_form;

    delete _export_audio_dialog;

    delete _transform_info_form;

    delete _playbackmarkers_controller;

    delete _about_dialog;

    delete _selection_controller;

    delete _navigation_controller;

    delete _playback_controller;

    delete _brush_controller;

    delete _record_controller;

    delete _comment_controller;

    delete _matlab_controller;

    delete _graph_controller;

    delete _tooltip_controller;

    delete _fantracker_controller;

    BOOST_ASSERT( _timeline_controller );
	delete _timeline_controller;

    BOOST_ASSERT( _timeline_view );
    delete _timeline_view;

    BOOST_ASSERT( _render_controller );
    _render_controller.reset();

    BOOST_ASSERT( _render_view );
    delete _render_view;
}


ToolFactory::
        ToolFactory()
            :
            render_model( 0 ),
            selection_model( 0 ),
            playback_model( 0 )
{
    BOOST_ASSERT( false );
}


} // namespace Tools
