#include "toolfactory.h"

// Tools
#include "timelinecontroller.h"
#include "timelineview.h"
#include "rendercontroller.h"
#include "renderview.h"
#include "selectioncontroller.h"
#include "selectionview.h"

// Sonic AWE
#include "sawe/project.h"
#include "ui/mainwindow.h"

// Qt
#include <QHBoxLayout>

namespace Tools
{

ToolFactory::
        ToolFactory(Sawe::Project* p)
:   render_model( p ),
    selection_model( p ),
    playback_model( &p->worker )
{
    _render_view = new RenderView(&render_model);
    _render_controller = new RenderController(_render_view);

    _timeline_view = new TimelineView(p, _render_view);
    _timeline_controller = new TimelineController(_timeline_view);

    _selection_view = new SelectionView(&selection_model);
    _selection_controller = new SelectionController(_selection_view, _render_view );
}


ToolFactory::
        ~ToolFactory()
{
    delete _render_controller;

    // TODO figure out a way to make sure that the rendering thread is not
    // doing anything with the views

    // The _render_view and _timeline_view widget are released by MainWindow
    // that owns the widget. This might happen both before and after this
    // destructor.
}

} // namespace Tools
