#ifndef COMMAND_H
#define COMMAND_H

#include <string>
#include <boost/shared_ptr.hpp>

namespace Tools {
namespace Commands {

/**
 * @brief The Command class represents user actions.
 *
 * The Command class is intended to be the interface towards changing the
 * user-level state of the application. Commands are invoked through
 * CommandInvoker::invokeCommand(pCommand). CommandInvoker keeps track of a
 * CommandList that describes the order of executed and undone commands.
 *
 * TODO should use QString toString() instead of std::string toString();
 */
class Command
{
public:
    Command();
    virtual ~Command();

    virtual std::string toString() = 0;

private:
    /**
     Only class CommandList calls Command methods.
     */
    friend class CommandList;


    /**
      @brief meldPrevCommand stores the undo-state of another Command.

      If two similiar Commands are executed right after each other they might
      be melded into one only command. This is typically used for small
      incremental changes within a short time interval.

      'meldPrevCommand' is only called right after this command is executed.
      This command will be executed regardless of the return value of meldPrevCommand.

      When a command is being redone, meldPrevCommand is not called.

      @return If meldPrevCommand returns true 'prev' will be removed from the
      command list. And calling this->undo() should restore the state that was
      prev->undo() would have restored.
      */
    virtual bool meldPrevCommand(Command* /*prev*/) { return false; }


    /**
      @brief addToList describes whether this Command should go into the
      command history listing.
      @return true if this command should go into the command history listing.
     */
    virtual bool addToList() { return true; }


    /**
      @brief execute performs this command. execute() is called when the
      command is invoked. execute() can also be called again if the command
      has been undone by calling undo().
     */
    virtual void execute() = 0;


    /**
      @brief undo restores the state that was changed by execute().
      @see execute()
     */
    virtual void undo() = 0;
};

typedef boost::shared_ptr<Command> pCommand;

} // namespace Commands
} // namespace Tools

#endif // COMMAND_H
