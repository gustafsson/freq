#include "commandinvoker.h"

#include <boost/foreach.hpp>

namespace Tools {
namespace Commands {

CommandInvoker::
        CommandInvoker(Sawe::Project * project)
            :
            project_(project)
{
}


CommandInvoker::
        ~CommandInvoker()
{
}


void CommandInvoker::
        invokeCommand(pCommand cmd)
{
    commandList_.invoke(cmd);
    emit projectChanged(commandList_.present());
}


void CommandInvoker::
        redo()
{
    commandList_.redo();
    emit projectChanged(commandList_.present());
}


void CommandInvoker::
        undo()
{
    commandList_.undo();
    emit projectChanged(commandList_.present());
}


const CommandList& CommandInvoker::
        commandList() const
{
    return commandList_;
}


Sawe::Project * CommandInvoker::
        project() const
{
    return project_;
}


} // namespace Commands
} // namespace Tools
