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
    _qx = model->_qx;
    _qy = model->_qy;
    _qz = model->_qz;
    _px = model->_px;
    _py = model->_py;
    _pz = model->_pz;
    _rx = model->_rx;
    _ry = model->_ry;
    _rz = model->_rz;
    xscale = model->xscale;
    zscale = model->zscale;
    orthoview = &model->orthoview;
}


void NavigationState::
        restoreState()
{
    model->_qx = _qx;
    model->_qy = _qy;
    model->_qz = _qz;
    model->_px = _px;
    model->_py = _py;
    model->_pz = _pz;
    model->_rx = _rx;
    model->_ry = _ry;
    model->_rz = _rz;
    model->xscale = xscale;
    model->zscale = zscale;
    model->orthoview = orthoview;
}


bool NavigationState::
        isSet()
{
    return _is_set;
}


} // namespace Commands
} // namespace Tools
