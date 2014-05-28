#include "recordedcommand.h"

#include "tools/rendermodel.h"

namespace Tools {
namespace Commands {

RecordedCommand::
        RecordedCommand(Adapters::Recorder::ptr recording,
                        Signal::IntervalType prevLength,
                        Tools::RenderModel* model,
                        Signal::Processing::IInvalidator::ptr iinvalidator)
            :
            recording(recording),
            model(model),
            iinvalidator(iinvalidator),
            undone(false),
            prev_qx(-1)
{
    const auto d = recording.raw ()->data ().read ();
    recordedData = d->samples.read(Signal::Interval(prevLength, d->samples.spannedInterval().last) );
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
        {
            auto d = recording.raw ()->data ().write ();
            recordedData->set_sample_offset ( (long long)d->samples.spannedInterval().count() );
            d->samples.put (recordedData);
        }

        iinvalidator.write ()->deprecateCache(recordedData->getInterval());

        if (0<=prev_qx)
            model->_qx = prev_qx;
    }
}


void RecordedCommand::
        undo()
{
    recording.raw ()->data ()->samples.invalidate_samples (recordedData->getInterval());
    prev_qx = model->_qx;
    iinvalidator.write ()->deprecateCache (recordedData->getInterval());
    if (prev_qx == model->_qx)
        prev_qx = -1;

    undone = true;
}

} // namespace Commands
} // namespace Tools
