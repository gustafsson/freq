#include "commandlist.h"

#include <boost/foreach.hpp>

namespace Tools {
namespace Commands {

CommandList::
        CommandList()
            :
            presentCommand(0)
{
}


CommandList::
        ~CommandList()
{

}


std::string CommandList::
        canUndo() const
{
    if (0 < presentCommand)
        return commands[presentCommand-1]->toString();
    return std::string();
}


void CommandList::
        undo()
{
    if (canUndo().empty())
        return;

    presentCommand--;

    commands[presentCommand]->undo();
}


std::string CommandList::
        canRedo() const
{
    if (presentCommand < commands.size())
        return commands[presentCommand]->toString();
    return std::string();
}


void CommandList::
        redo()
{
    if (canRedo().empty())
        return;

    presentCommand++;

    commands[presentCommand-1]->execute();
}


const Command* CommandList::
        present() const
{
    if (0 == presentCommand)
        return 0;

    return commands[presentCommand-1].get();
}


const std::vector<CommandP>& CommandList::
        getCommandList() const
{
    return commands;
}


void CommandList::
        execute( CommandP p)
{
    bool melded = false;
    if ( 0 < presentCommand )
        melded = p->meldPrevCommand( commands[ presentCommand-1 ].get() );

    // discard any posibilities to 'redo'
    commands.resize( presentCommand );

    if (melded)
    {
        // replace previous command
        commands[ presentCommand-1 ] = p;
    }
    else
    {
        // append a new command and increase 'presentCommand'
        commands.push_back( p );
        presentCommand++;
    }

    commands[presentCommand-1]->execute();
}


std::string CommandList::
        toString() const
{
    std::string description;
    BOOST_FOREACH( CommandP p, commands )
    {
        if (!description.empty())
            description += "\n";
        description += p->toString();
    }
    return description;
}

} // namespace Commands
} // namespace Tools
