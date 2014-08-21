#include "recorderoperation.h"

namespace Signal {

MicrophoneRecorderOperation::
        MicrophoneRecorderOperation( Signal::Recorder::ptr recorder )
    :
      recorder_(recorder)
{
}


Signal::pBuffer MicrophoneRecorderOperation::
        process(Signal::pBuffer b)
{
    return recorder_.write ()->read (b->getInterval ());
}


MicrophoneRecorderDesc::
        MicrophoneRecorderDesc(Signal::Recorder::ptr recorder, Signal::Recorder::IGotDataCallback::ptr invalidator)
    :
      recorder_(recorder)
{
    recorder_.write ()->setDataCallback(invalidator);
}


void MicrophoneRecorderDesc::
        startRecording()
{
    recorder_.write ()->startRecording ();
}


void MicrophoneRecorderDesc::
        stopRecording()
{
    recorder_.write ()->stopRecording ();
}


bool MicrophoneRecorderDesc::
        isStopped()
{
    return recorder_.write ()->isStopped ();
}


bool MicrophoneRecorderDesc::
        canRecord()
{
    return recorder_.write ()->canRecord ();
}


Signal::Interval MicrophoneRecorderDesc::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Signal::Interval MicrophoneRecorderDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    return I;
}


Signal::OperationDesc::ptr MicrophoneRecorderDesc::
        copy() const
{
    EXCEPTION_ASSERTX(false, "Can't make a copy of microphone recording");
    return Signal::OperationDesc::ptr();
}


Signal::Operation::ptr MicrophoneRecorderDesc::
        createOperation(Signal::ComputingEngine*) const
{
    // any computing engine can read already recorded data, i.e. storing
    // recorded data is done by another thread
    Signal::Operation::ptr r(new MicrophoneRecorderOperation(recorder_));
    return r;
}


MicrophoneRecorderDesc::Extent MicrophoneRecorderDesc::
        extent() const
{
    auto data = recorder_.raw ()->data ();
    float fs = data.raw ()->sample_rate;;
    float L = recorder_.read ()->length();

    MicrophoneRecorderDesc::Extent x;
    x.interval = Signal::Interval(0, fs*L);
    x.number_of_channels = data.raw ()->num_channels;
    x.sample_rate = fs;

    return x;
}


Signal::Recorder::ptr MicrophoneRecorderDesc::
        recorder() const
{
    return recorder_;
}


} // namespace Signal
