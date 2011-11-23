#ifndef NAVIGATIONCOMMAND_H
#define NAVIGATIONCOMMAND_H

#include "viewcommand.h"

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

    float _qx, _qy, _qz;
    float _px, _py, _pz,
        _rx, _ry, _rz;
    float xscale;
    float zscale;
    float orthoview;
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
    virtual void execute();
    virtual void undo();

    NavigationState prevState, newState;
};

} // namespace Commands
} // namespace Tools

#endif // NAVIGATIONCOMMAND_H
