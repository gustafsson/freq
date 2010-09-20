#include "toolfactory.h"
#include "timelinecontroller.h"
#include "timelineview.h"
#include "timelinemodel.h"
#include "rendercontroller.h"

namespace Tools
{

ToolFactory::
        ToolFactory(Sawe::Project* p)
:   render_model(p),
    selection_model(p),

    render_view(&render_model),
    selection_view(&selection_model)
{
    _render_controller = new RenderController(&render_view);

    //_timeline_model = new TimelineModel();
    //_timeline_view.reset( new TimelineView(p, render_view.displayWidget));
    //_timeline_controller = new TimelineController(view);
}


ToolFactory::
        ~ToolFactory()
{
    // TODO figure out a way to make sure that the rendering thread is not
    // doing anything with the views
}


} // namespace Tools
