#ifndef ROTATECAMERACOMMAND_H
#define ROTATECAMERACOMMAND_H

#include "navigationcommand.h"

namespace Tools {
    class RenderModel;

namespace Commands {

class RotateCameraCommand : public Tools::Commands::NavigationCommand
{
public:
    RotateCameraCommand(Tools::RenderModel* model, float dt, float ds);

    virtual std::string toString();
    virtual void executeFirst();

private:
    float dx, dy;
};

} // namespace Commands
} // namespace Tools

#endif // ROTATECAMERACOMMAND_H
