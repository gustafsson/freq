#ifndef SIGNAL_RECORDEROPERATION_H
#define SIGNAL_RECORDEROPERATION_H

#include "recorder.h"
#include "operation.h"

namespace Signal {

/**
 * @brief The MicrophoneRecorderOperation class should provide access to recorded data.
 */
class MicrophoneRecorderOperation: public Signal::Operation
{
public:
    MicrophoneRecorderOperation( Recorder::ptr recorder );

    virtual Signal::pBuffer process(Signal::pBuffer b);

private:
    Recorder::ptr recorder_;
};


/**
 * @brief The MicrophoneRecorderDesc class should control the behaviour of a recording.
 */
class MicrophoneRecorderDesc: public Signal::OperationDesc
{
public:
    MicrophoneRecorderDesc( Recorder::ptr, Recorder::IGotDataCallback::ptr invalidator );

    void startRecording();
    void stopRecording();
    bool isStopped() const;
    bool canRecord() const;
    Recorder::ptr recorder() const;

    // OperationDesc
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    virtual Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    virtual OperationDesc::ptr copy() const;
    virtual Signal::Operation::ptr createOperation( Signal::ComputingEngine* engine ) const;
    virtual Extent extent() const;

private:
    Recorder::ptr recorder_;
};

} // namespace Signal

#endif // SIGNAL_RECORDEROPERATION_H
