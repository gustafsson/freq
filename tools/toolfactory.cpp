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
:   render_model( p ),
    selection_model( p ),
    playback_model( &selection_model )
{
    _render_view = new RenderView(&render_model);
    _render_controller.reset( new RenderController(_render_view) );

    _timeline_view = new TimelineView(p, _render_view);
    _timeline_controller = new TimelineController(_timeline_view);

    _selection_controller = new SelectionController(&selection_model, _render_view );

    _navigation_controller = new NavigationController(_render_view);

    _playback_view.reset( new PlaybackView(&playback_model, _render_view) );
    _playback_controller = new PlaybackController(p, _playback_view.data(), _render_view);

    _brush_model.reset( new BrushModel(p, &render_model) );
    _brush_view.reset( new BrushView(_brush_model.data() ));
    _brush_controller = new BrushController( _brush_view.data(), _render_view );

    if (RecordModel::canCreateRecordModel(p))
    {
        _record_model.reset( new RecordModel(p, _render_view) );
        _record_view.reset( new RecordView(_record_model.data() ));
        _record_controller = new RecordController( _record_view.data() );
    }

    _comment_controller = new CommentController( _render_view );

    _matlab_controller = new MatlabController( p, _render_view );

    _graph_controller = new GraphController( _render_view );

    _tooltip_model.reset( new TooltipModel() );
    _tooltip_view.reset( new TooltipView(_tooltip_model.data(), _render_view ));
    _tooltip_controller = new TooltipController(
            _tooltip_view.data(), _render_view,
            dynamic_cast<CommentController*>(_comment_controller.data()) );

    _about_dialog = new AboutDialog( p );

    _playbackmarkers_model.reset( new PlaybackMarkersModel() );
    _playbackmarkers_view.reset( new PlaybackMarkersView( _playbackmarkers_model.data() ));
    _playbackmarkers_controller = new PlaybackMarkersController(
            _playbackmarkers_view.data(), _render_view );
}


ToolFactory::
        ~ToolFactory()
{
    TaskInfo ti(__FUNCTION__);
    // Try to clear things in the opposite order that they were created

    delete _about_dialog;

    if (!_selection_controller.isNull())
        delete _selection_controller;

    if (!_navigation_controller .isNull())
        delete _navigation_controller;

    if (!_playback_controller.isNull())
        delete _playback_controller;

    if (!_brush_controller.isNull())
        delete _brush_controller;

    if (!_record_controller.isNull())
        delete _record_controller;

    if (!_comment_controller.isNull())
        delete _comment_controller;

    if (!_matlab_controller.isNull())
        delete _matlab_controller;

    if (!_graph_controller.isNull())
        delete _graph_controller;

    if (!_tooltip_controller.isNull())
        delete _tooltip_controller;

    BOOST_ASSERT( _timeline_controller );
	delete _timeline_controller;

    BOOST_ASSERT( _timeline_view );
    delete _timeline_view;

    BOOST_ASSERT( _render_controller );
    _render_controller.reset();

    BOOST_ASSERT( _render_view );
    delete _render_view;
}

} // namespace Tools
