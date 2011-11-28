#ifndef CHANGESELECTIONCOMMAND_H
#define CHANGESELECTIONCOMMAND_H

#include "command.h"
#include "signal/operation.h"

namespace Tools {
    class SelectionController;

namespace Commands {

class ChangeSelectionCommand : public Tools::Commands::Command
{
public:
    ChangeSelectionCommand(Tools::SelectionController*p, Signal::pOperation s);

    virtual void execute();
    virtual void undo();
    virtual std::string toString();

private:
    Signal::pOperation s;
    Signal::pOperation prevSelection;
    Tools::SelectionController* p;
};

} // namespace Commands
} // namespace Tools

#endif // CHANGESELECTIONCOMMAND_H
