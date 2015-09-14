#ifndef NAVIGATIONCOMMAND_H
#define NAVIGATIONCOMMAND_H

#include "viewcommand.h"
#include "GLvector.h"
#include "tools/support/rendercamera.h"

namespace Tools {
    class RenderModel;
namespace Commands {

class NavigationState
{
public:
    NavigationState(Tools::RenderModel* model);

    void storeState();
    void restoreState();
    bool isSet();

private:
    bool _is_set;

    Tools::Support::RenderCamera camera;
    Tools::RenderModel* model;
};


/**
The purpose of class NavigationCommand is for different subclasses of NavigationCommands to
recognize eachother and replace eachother even if they are not subsequent. Because they all
share the same undo state.
*/
class NavigationCommand : public Tools::Commands::ViewCommand
{
protected:
    NavigationCommand(Tools::RenderModel* model);

    virtual void executeFirst() = 0;
    Tools::RenderModel* model;

private:
    virtual bool meldPrevCommand(Command*);
    virtual bool addToList() { return false; }
    virtual void execute();
    virtual void undo();

    NavigationState prevState, newState;
};

} // namespace Commands
} // namespace Tools

#endif // NAVIGATIONCOMMAND_H
