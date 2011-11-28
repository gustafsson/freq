#include "rotatecameracommand.h"

#include "tools/rendermodel.h"

namespace Tools {
namespace Commands {

RotateCameraCommand::
        RotateCameraCommand(Tools::RenderModel* model, float dx, float dy)
            :
            NavigationCommand(model),
            dx(dx),
            dy(dy)
{
}


std::string RotateCameraCommand::
        toString()
{
    return "Rotate camera";
}


void RotateCameraCommand::
        executeFirst()
{
    float rs = 0.2;

    model->_ry += (1-model->orthoview)*rs * dx;
    model->_rx += rs * dy;
    if (model->_rx<10) model->_rx=10;
    if (model->_rx>90) { model->_rx=90; model->orthoview=1; }
    if (0<model->orthoview && model->_rx<90) { model->_rx=90; model->orthoview=0; }
}


} // namespace Commands
} // namespace Tools
