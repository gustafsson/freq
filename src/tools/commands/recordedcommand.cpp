#include "recordedcommand.h"

#include "tools/rendermodel.h"

namespace Tools {
namespace Commands {

RecordedCommand::
        RecordedCommand(Adapters::Recorder::Ptr recording,
                        Signal::IntervalType prevLength,
                        Tools::RenderModel* model,
                        Signal::Processing::IInvalidator::Ptr iinvalidator)
            :
            recording(recording),
            model(model),
            iinvalidator(iinvalidator),
            undone(false),
            prev_qx(-1)
{
    Adapters::Recorder::WritePtr r(recording);
    recordedData = r->data().read(Signal::Interval(prevLength, r->number_of_samples()) );
}


std::string RecordedCommand::
        toString()
{
    return "Recorded " + Signal::SourceBase::lengthLongFormat( recordedData->length() );
}


void RecordedCommand::
        execute()
{
    if (undone)
    {
        Adapters::Recorder::WritePtr r(recording);
        recordedData->set_sample_offset( r->number_of_samples() );
        r->data().put(recordedData);
        write1(iinvalidator)->deprecateCache(recordedData->getInterval());

        if (0<=prev_qx)
            model->_qx = prev_qx;
    }
}


void RecordedCommand::
        undo()
{
    write1(recording)->data().invalidate_samples(recordedData->getInterval());
    prev_qx = model->_qx;
    write1(iinvalidator)->deprecateCache(recordedData->getInterval());
    if (prev_qx == model->_qx)
        prev_qx = -1;

    undone = true;
}

} // namespace Commands
} // namespace Tools
