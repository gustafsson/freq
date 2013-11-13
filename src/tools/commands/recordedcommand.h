#ifndef RECORDEDCOMMAND_H
#define RECORDEDCOMMAND_H

#include "operationcommand.h"
#include "signal/source.h"

namespace Signal {
    class DeprecatedOperation;
}

namespace Tools {
    class RenderModel;

namespace Commands {

class RecordedCommand : public Tools::Commands::OperationCommand
{
public:
    RecordedCommand(Signal::DeprecatedOperation* recording, Signal::IntervalType prevLength, Tools::RenderModel* model);

    virtual std::string toString();
    virtual void execute();
    virtual void undo();

private:
    Signal::DeprecatedOperation* recording;
    Signal::pBuffer recordedData;
    Tools::RenderModel* model;
    bool undone;
    float prev_qx;
};

} // namespace Commands
} // namespace Tools

#endif // RECORDEDCOMMAND_H
