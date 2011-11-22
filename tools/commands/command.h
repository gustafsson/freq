#ifndef COMMAND_H
#define COMMAND_H

#include <string>
#include <boost/shared_ptr.hpp>

namespace Tools {
namespace Commands {

class Command
{
public:
    Command();
    virtual ~Command();

    virtual std::string toString() = 0;

private:
    friend class CommandList;

    /**
      If two similiar Commands are executed right after each other they
      might be melded into one only command.

      Typically used for small incremental changes within a short time interval.

      meldPrevCommand is only called right before this command is executed.
      This command will be executed regardless of the return value of meldPrevCommand.

      When a command is being redone, meldPrevCommand is not called.
      */
    virtual bool meldPrevCommand(Command*) { return false; }


    virtual void execute() = 0;
    virtual void undo() = 0;
};

typedef boost::shared_ptr<Command> CommandP;

} // namespace Commands
} // namespace Tools

#endif // COMMAND_H
