#include "changetoolcommand.h"

#include "tools/support/toolselector.h"
#include "demangle.h"

#include <QWidget>

namespace Tools {
namespace Commands {

ChangeToolCommand::
        ChangeToolCommand(QWidget* newTool, Support::ToolSelector* selector)
            :
            newTool(newTool),
            prevTool(0),
            selector(selector)
{
}


void ChangeToolCommand::
        execute()
{
    prevTool = this->selector->currentTool();
    selector->setCurrentToolCommand( newTool );
}


void ChangeToolCommand::
        undo()
{
    selector->setCurrentToolCommand( prevTool );
}


std::string ChangeToolCommand::
        toString()
{
    return newTool ? vartype(*newTool) : "No tool in " + vartype(*selector->parentTool());
}

} // namespace Commands
} // namespace Tools
