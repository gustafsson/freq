#include "toolfactory.h"

namespace Tools
{

ToolFactory::
        ToolFactory(Sawe::Project* p)
:   render_model(p),
    selection_model(p),

    render_view(&render_model),
    selection_view(&selection_model)
{}


ToolFactory::
        ~ToolFactory()
{
    // TODO figure out a way to make sure that the rendering thread is not
    // doing anything with the views
}


} // namespace Tools
