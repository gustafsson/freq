#ifndef APPENDOPERATIONCOMMAND_H
#define APPENDOPERATIONCOMMAND_H

#include "operationcommand.h"
#include "signal/operation.h"

namespace Sawe {
    class Project;
}

namespace Tools {
namespace Commands {

class AppendOperationCommand : public Tools::Commands::OperationCommand
{
public:
    AppendOperationCommand(Sawe::Project*p, Signal::pOperation s);

    virtual void execute();
    virtual void undo();
    virtual std::string toString();

private:
    Signal::pOperation s;
    Signal::pOperation prevHead, newHead;
    Sawe::Project* p;
};

} // namespace Commands
} // namespace Tools

#endif // APPENDOPERATIONCOMMAND_H
