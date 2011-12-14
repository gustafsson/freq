#include "recordedcommand.h"

#include "tools/rendermodel.h"
#include "adapters/microphonerecorder.h"

namespace Tools {
namespace Commands {

RecordedCommand::
        RecordedCommand(Signal::Operation* recording, Signal::IntervalType prevLength, Tools::RenderModel* model)
            :
            recording(recording),
            model(model),
            undone(false),
            prev_qx(-1)
{
    Adapters::MicrophoneRecorder* r = dynamic_cast<Adapters::MicrophoneRecorder*>(recording);
    recordedData = r->data().readAllChannelsFixedLength(Signal::Interval(prevLength, recording->number_of_samples()) );
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
        Adapters::MicrophoneRecorder* r = dynamic_cast<Adapters::MicrophoneRecorder*>(recording);
        recordedData->sample_offset = r->number_of_samples();
        r->data().put(recordedData);
        r->invalidate_samples(recordedData->getInterval());

        if (0<=prev_qx)
            model->_qx = prev_qx;
    }
}


void RecordedCommand::
        undo()
{
    Adapters::MicrophoneRecorder* r = dynamic_cast<Adapters::MicrophoneRecorder*>(recording);
    r->data().invalidate_and_forget_samples(recordedData->getInterval());
    prev_qx = model->_qx;
    r->invalidate_samples(recordedData->getInterval());
    if (prev_qx == model->_qx)
        prev_qx = -1;

    undone = true;
}

} // namespace Commands
} // namespace Tools
