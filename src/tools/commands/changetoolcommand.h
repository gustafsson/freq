#ifndef CHANGETOOLCOMMAND_H
#define CHANGETOOLCOMMAND_H

#include "command.h"

class QWidget;

namespace Tools {
    namespace Support {
        class ToolSelector;
    }

namespace Commands {

class ChangeToolCommand : public Tools::Commands::Command
{
public:
    ChangeToolCommand(QWidget* newTool, Support::ToolSelector* selector);

    virtual void execute();
    virtual void undo();
    virtual std::string toString();

private:
    virtual bool addToList() { return false; }
    QWidget* newTool, *prevTool;
    Support::ToolSelector* selector;
};

} // namespace Commands
} // namespace Tools

#endif // CHANGETOOLCOMMAND_H
