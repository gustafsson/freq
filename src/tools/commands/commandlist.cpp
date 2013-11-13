#include "commandlist.h"

#include <boost/foreach.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::gregorian;
using namespace boost::posix_time;

namespace Tools {
namespace Commands {

CommandList::
        CommandList()
            :
            presentCommand(0),
            lastCommandTimeStamp(0)
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


const std::vector<pCommand>& CommandList::
        getCommandList() const
{
    return commands;
}


void CommandList::
        invoke( pCommand p)
{
    if (!p->addToList())
    {
        p->execute();
        return;
    }

    // discard any posibilities to 'redo' after 'presentCommand'
    commands.resize( presentCommand );

    // append a new command and increase 'presentCommand'
    commands.push_back( p );
    presentCommand++;

    // execute command
    commands[presentCommand-1]->execute();

    // check if the two most recent commands can be melded into one
    tryMeld();
}


void CommandList::
        tryMeld()
{
    ptime epoch(date(2011,boost::date_time::Jan,1));
    ptime local_time(microsec_clock::local_time());
    unsigned now = (local_time - epoch).ticks() & 0xFFFFFFFF;

    time_duration duration(0,0,0,now-lastCommandTimeStamp);
    if ( 1 < presentCommand && duration.total_milliseconds() < 1000 )
    {
        if (commands[presentCommand-1]->meldPrevCommand( commands[ presentCommand-2 ].get() ))
        {
            // replace previous command
            presentCommand--;
            commands[presentCommand-1] = commands[ presentCommand ];
            commands.resize(presentCommand);
        }
    }

    lastCommandTimeStamp = now;
}


std::string CommandList::
        toString() const
{
    std::string description;
    BOOST_FOREACH( pCommand p, commands )
    {
        if (!description.empty())
            description += "\n";
        description += p->toString();
    }
    return description;
}

} // namespace Commands
} // namespace Tools
