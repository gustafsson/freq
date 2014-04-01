#ifndef TOOLS_COMMANDS_APPENDOPERATIONDESCCOMMAND_H
#define TOOLS_COMMANDS_APPENDOPERATIONDESCCOMMAND_H

#include "operationcommand.h"
#include "signal/processing/chain.h"
#include "signal/processing/targetneeds.h"
#include "signal/operation.h"

namespace Tools {
namespace Commands {

/**
 * @brief The AppendOperationDescCommand class should add a new operation to
 * the signal processing chain at the given targets current position.
 */
class AppendOperationDescCommand : public Tools::Commands::OperationCommand
{
public:
    AppendOperationDescCommand(
            Signal::OperationDesc::ptr o,
            Signal::Processing::Chain::ptr c,
            Signal::Processing::TargetMarker::ptr at);

    virtual void execute();
    virtual void undo();
    virtual std::string toString();

private:
    Signal::OperationDesc::ptr operation_;
    Signal::Processing::Chain::ptr chain_;
    Signal::Processing::TargetMarker::ptr at_;

public:
    static void test();
};

} // namespace Commands
} // namespace Tools

#endif // TOOLS_COMMANDS_APPENDOPERATIONDESCCOMMAND_H
