#ifndef COMMANDLIST_H
#define COMMANDLIST_H

#include <vector>

#include "command.h"

namespace Tools {
namespace Commands {

class CommandList
{
public:
    CommandList();
    virtual ~CommandList();

    void undo();
    void redo();
    void invoke( pCommand );
    std::string toString() const;

    std::string canUndo() const;
    std::string canRedo() const;

    const Command* present() const;
    const std::vector<pCommand>& getCommandList() const;

private:
    /**
      presentCommand points to the index in commands where the next operation will be executed.
      */
    unsigned presentCommand;
    unsigned lastCommandTimeStamp;
    std::vector<pCommand> commands;

    void tryMeld();
};

} // namespace Commands
} // namespace Tools

#endif // COMMANDLIST_H
