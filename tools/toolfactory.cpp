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

    _brush_model.reset( new BrushModel(p) );
    _brush_view.reset( new BrushView(_brush_model.data() ));
    _brush_controller = new BrushController( _brush_view.data(), _render_view );

    if (RecordModel::canCreateRecordModel(p))
    {
        _record_model.reset( new RecordModel(p) );
        _record_view.reset( new RecordView(_record_model.data() ));
        _record_controller = new RecordController( _record_view.data(), _render_view );
    }

    _comment_controller = new CommentController( _render_view );
}


ToolFactory::
        ~ToolFactory()
{
    TaskTimer(__FUNCTION__).suppressTiming();

    if (!_navigation_controller .isNull())
        delete _navigation_controller;

    if (!_selection_controller.isNull())
        delete _selection_controller;

    if (!_playback_controller.isNull())
        delete _playback_controller;

    // The _render_view and _timeline_view widget are released by MainWindow
    // that owns the widget. This might happen both before and after this
    // destructor.
}

} // namespace Tools
