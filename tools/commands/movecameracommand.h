#ifndef MOVECAMERACOMMAND_H
#define MOVECAMERACOMMAND_H

#include "navigationcommand.h"

namespace Tools {
    class RenderModel;
namespace Commands {

class MoveCameraCommand : public Tools::Commands::NavigationCommand
{
public:
    MoveCameraCommand(Tools::RenderModel* model, float dt, float ds);

    virtual std::string toString();
    virtual void executeFirst();

private:
    float dt, ds;
};

} // namespace Commands
} // namespace Tools

#endif // MOVECAMERACOMMAND_H
