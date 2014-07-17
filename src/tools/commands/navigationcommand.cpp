#include "navigationcommand.h"

#include "tools/rendermodel.h"

namespace Tools {
namespace Commands {


NavigationCommand::
        NavigationCommand(Tools::RenderModel* model)
            :
            model(model),
            prevState(model),
            newState(model)
{
}


bool NavigationCommand::
        meldPrevCommand(Command* prevCommand)
{
    NavigationCommand* prevNavigation = dynamic_cast<NavigationCommand*>(prevCommand);
    if (!prevNavigation)
        return false;

    prevState = prevNavigation->prevState;

    return true;
}


void NavigationCommand::
        execute()
{
    if (newState.isSet())
        newState.restoreState();
    else
    {
        prevState.storeState();
        executeFirst();
        newState.storeState();
    }
}


void NavigationCommand::
        undo()
{
    prevState.restoreState();
}



NavigationState::
        NavigationState(Tools::RenderModel* model)
            :
            _is_set(false),
            model(model)
{

}


void NavigationState::
        storeState()
{
    _is_set = true;
    camera = model->camera;
    camera.orthoview.reset (&model->camera.orthoview);
}


void NavigationState::
        restoreState()
{
    model->camera = camera;
    model->camera.orthoview.reset (&camera.orthoview);
}


bool NavigationState::
        isSet()
{
    return _is_set;
}


} // namespace Commands
} // namespace Tools
