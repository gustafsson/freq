#ifndef RECORDER_H
#define RECORDER_H

#include "cache.h"
#include "shared_state.h"
#include "verifyexecutiontime.h"
#include <QMutex>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace Adapters {

class Recorder
{
public:
    typedef shared_state<Recorder> ptr;

    struct shared_state_traits: shared_state_traits_default {
        double timeout() { return 0.500; }

        void was_locked() {
            verify_ = VerifyExecutionTime::start(0.250);
        }

        void was_unlocked() {
            verify_.reset();
        }

    private:
        int verify_execution_time_ms() { return 250; }
        VerifyExecutionTime::ptr verify_;
    };


    class IGotDataCallback
    {
    public:
        typedef shared_state<IGotDataCallback> ptr;

        virtual ~IGotDataCallback() {}

        virtual void markNewlyRecordedData(Signal::Interval what)=0;
    };


    Recorder();
    virtual ~Recorder();

    virtual void startRecording() = 0;
    virtual void stopRecording() = 0;
    virtual bool isStopped() const = 0;
    virtual bool canRecord() = 0;

    float time_since_last_update();
    void setDataCallback( IGotDataCallback::ptr invalidator );
    Signal::Cache& data() { return _data; }

    // virtual from Signal::SourceBase
    virtual std::string name() = 0;
    virtual float sample_rate() const = 0;
    virtual unsigned num_channels() const = 0;

    // overloaded from Signal::FinalSource
    virtual Signal::pBuffer read( const Signal::Interval& I );
    virtual Signal::IntervalType number_of_samples() const;
    virtual float length() const;

protected:
    mutable QMutex _data_lock;
    Signal::Cache _data; // TODO use shared_state<Signal::Cache>
    IGotDataCallback::ptr _invalidator;
    std::exception_ptr _exception;
    float _offset;
    boost::posix_time::ptime _start_recording, _last_update;
    Signal::IntervalType actual_number_of_samples() const;

    virtual float time() const;
};


} // namespace Adapters

#endif // RECORDER_H
