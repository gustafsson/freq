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
    double l = model->tfr_mapping().read ()->length();

    vectord& q = model->camera.q;
    q[0] += dt;
    q[2] += ds;

    if (q[0]<0) q[0]=0;
    if (q[2]<0) q[2]=0;
    if (q[2]>1) q[2]=1;
    if (q[0]>l) q[0]=l;
}


} // namespace Commands
} // namespace Tools
