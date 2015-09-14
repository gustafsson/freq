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

    auto c = model->camera.write ();
    vectord& r = c->r;

    r[1] += (1-c->orthoview)*rs * dx;
    r[0] += rs * dy;
    if (r[0]<0) { r[0]=0; c->orthoview=1; }
    if (r[0]>90) { r[0]=90; c->orthoview=1; }
    if (0<c->orthoview && r[0]<90 && r[0]>=45) { c->orthoview=0; }
    if (0<c->orthoview && r[0]<45 && r[0]>0) { c->orthoview=0; }
}


} // namespace Commands
} // namespace Tools
