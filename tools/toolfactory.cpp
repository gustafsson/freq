#include "toolfactory.h"
#include "timelinecontroller.h"
#include "timelineview.h"
#include "rendercontroller.h"
#include "renderview.h"

namespace Tools
{

ToolFactory::
        ToolFactory(Sawe::Project* p)
:   render_model(p),
    selection_model(p),

    selection_view(&selection_model)
{
    _render_view = new RenderView(&render_model);
    _render_controller = new RenderController(_render_view);

    _timeline_view = new TimelineView(p, _render_view);
    _timeline_controller = new TimelineController(_timeline_view);
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
