#include "projectstate.h"

#include <boost/foreach.hpp>

namespace Tools {
namespace Commands {

ProjectState::
        ProjectState(Sawe::Project * project)
            :
            project_(project)
{
}


ProjectState::
        ~ProjectState()
{
}


void ProjectState::
        executeCommand(CommandP cmd)
{
    commandList_.execute(cmd);
    emit projectChanged(commandList_.present());
}


void ProjectState::
        redo()
{
    commandList_.redo();
    emit projectChanged(commandList_.present());
}


void ProjectState::
        undo()
{
    commandList_.undo();
    emit projectChanged(commandList_.present());
}


const CommandList& ProjectState::
        commandList() const
{
    return commandList_;
}


Sawe::Project * ProjectState::
        project() const
{
    return project_;
}


} // namespace Commands
} // namespace Tools
