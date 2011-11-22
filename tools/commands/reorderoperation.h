#ifndef REORDEROPERATION_H
#define REORDEROPERATION_H

#include "signal/operation.h"

#include "operationcommand.h"

namespace Tools {
namespace Commands {

class ReorderOperation : public Tools::Commands::OperationCommand
{
public:
    ReorderOperation(Signal::pOperation operation, Signal::pOperation newSource);

    virtual void execute();
    virtual void undo();
    virtual std::string toString();

private:
    Signal::pOperation operation;
    Signal::pOperation newSource;
    Signal::pOperation oldSource;
};

} // namespace Commands
} // namespace Tools

#endif // REORDEROPERATION_H
