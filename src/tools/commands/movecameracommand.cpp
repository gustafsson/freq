#include "movecameracommand.h"

#include "tools/rendermodel.h"
#include "sawe/project.h"

namespace Tools {
namespace Commands {

MoveCameraCommand::
        MoveCameraCommand(Tools::RenderModel* model, float dt, float ds)
            :
            NavigationCommand(model),
            dt(dt),
            ds(ds)
{
}


std::string MoveCameraCommand::
        toString()
{
    return "Move camera";
}


void MoveCameraCommand::
        executeFirst()
{
    float l = model->project()->worker.source()->length();

    model->_qx += dt;
    model->_qz += ds;

    if (model->_qx<0) model->_qx=0;
    if (model->_qz<0) model->_qz=0;
    if (model->_qz>1) model->_qz=1;
    if (model->_qx>l) model->_qx=l;
}


} // namespace Commands
} // namespace Tools
