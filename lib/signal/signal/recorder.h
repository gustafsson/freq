#ifndef SIGNAL_RECORDER_H
#define SIGNAL_RECORDER_H

#include "signal/cache.h"
#include "shared_state.h"
#include "verifyexecutiontime.h"
#include "timer.h"

namespace Signal {

class Recorder
{
public:
    typedef shared_state<Recorder> ptr;
    typedef shared_state<const Recorder> const_ptr;

    struct shared_state_traits: shared_state_traits_default {
        double timeout() { return 0.500; }

        template<class T>
        void locked (T*) {
            verify_ = VerifyExecutionTime::start(0.250);
        }

        template<class T>
        void unlocked (T*) {
            verify_.reset ();
        }

    private:
        VerifyExecutionTime::ptr verify_;
    };


    // TODO use Signal::Processing::IInvalidator
    class IGotDataCallback
    {
    public:
        typedef shared_state<IGotDataCallback> ptr;

        virtual ~IGotDataCallback() {}

        virtual void markNewlyRecordedData(Signal::Interval what)=0;
    };


    struct Data
    {
        Data(float sample_rate, unsigned num_channels)
            : sample_rate (sample_rate),
              num_channels (num_channels)
        {}

        Signal::Cache samples;
        const float sample_rate;
        const unsigned num_channels;
    };


    Recorder();
    virtual ~Recorder();

    virtual void startRecording() = 0;
    virtual void stopRecording() = 0;
    virtual bool isStopped() const = 0;
    virtual bool canRecord() = 0;
    virtual std::string name() = 0;
    virtual float length();

    float time_since_last_update();
    void setDataCallback( IGotDataCallback::ptr invalidator );

    // Data race free and lock free methods
    shared_state<Data> data() { return _data; }
    shared_state<const Data> data() const { return _data; }
    float sample_rate() const;
    unsigned num_channels() const;

    // Data race free methods
    Signal::IntervalType number_of_samples() const;
    Signal::pBuffer read( const Signal::Interval& I );

protected:
    shared_state<Data> _data;
    float _offset;
    Timer _start_recording, _last_update;
    IGotDataCallback::ptr _invalidator;
    std::exception_ptr _exception;

    virtual float time();

    Signal::IntervalType actual_number_of_samples() const;
};


} // namespace Signal

#endif // SIGNAL_RECORDER_H
