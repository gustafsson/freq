#ifndef RECORDEDCOMMAND_H
#define RECORDEDCOMMAND_H

#include "operationcommand.h"
#include "signal/source.h"
#include "signal/operation.h"
#include "signal/processing/iinvalidator.h"
#include "adapters/recorder.h"

namespace Tools {
    class RenderModel;

namespace Commands {

class RecordedCommand : public Tools::Commands::OperationCommand
{
public:
    RecordedCommand(Adapters::Recorder::ptr recording, Signal::IntervalType prevLength, Tools::RenderModel* model, Signal::Processing::IInvalidator::ptr iinvalidator);

    virtual std::string toString();
    virtual void execute();
    virtual void undo();

private:
    Adapters::Recorder::ptr recording;
    // To do, should be another structure that can handle larger amounts of data.
    Signal::pBuffer recordedData;
    Tools::RenderModel* model;
    Signal::Processing::IInvalidator::ptr iinvalidator;
    bool undone;
    float prev_qx;
};

} // namespace Commands
} // namespace Tools

#endif // RECORDEDCOMMAND_H
