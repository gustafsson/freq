#ifndef REORDEROPERATION_H
#define REORDEROPERATION_H

#include "signal/operation.h"

#include "operationcommand.h"

namespace Tools {
namespace Commands {

class ReorderOperation : public Tools::Commands::OperationCommand
{
public:
    /**
      operation and operationTail is the same unless a sequence of operations should be moved as one
      */
    ReorderOperation(Signal::pOperation operation, Signal::pOperation operationTail, Signal::pOperation newSource);

    virtual void execute();
    virtual void undo();
    virtual std::string toString();

private:
    Signal::pOperation operation, operationTail;
    Signal::pOperation newSource, oldSource;
};

} // namespace Commands
} // namespace Tools

#endif // REORDEROPERATION_H
