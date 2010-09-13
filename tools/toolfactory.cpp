#include "toolfactory.h"

namespace Tools
{

ToolFactory::
        ToolFactory(pProject p)
:   render_model(p),
    selection_model(p),

    render_view(&selection_model)
    selection_view(&selection_model)
{}


ToolFactory::
        ~ToolFactory()
{
    // TODO figure out a way to make sure that the rendering thread is not
    // doing anything with the views
}


} // namespace Tools
