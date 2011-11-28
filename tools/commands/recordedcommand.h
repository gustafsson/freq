#ifndef RECORDEDCOMMAND_H
#define RECORDEDCOMMAND_H

#include "operationcommand.h"
#include "signal/source.h"

namespace Signal {
    class Operation;
}

namespace Tools {
    class RenderModel;

namespace Commands {

class RecordedCommand : public Tools::Commands::OperationCommand
{
public:
    RecordedCommand(Signal::Operation* recording, Signal::IntervalType prevLength, Tools::RenderModel* model);

    virtual std::string toString();
    virtual void execute();
    virtual void undo();

private:
    Signal::Operation* recording;
    Signal::pBuffer recordedData;
    Tools::RenderModel* model;
    bool undone;
    float prev_qx;
};

} // namespace Commands
} // namespace Tools

#endif // RECORDEDCOMMAND_H
