#ifndef RECORDER_H
#define RECORDER_H

#include "cache.h"
#include "volatileptr.h"
#include <QMutex>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace Adapters {

class Recorder: public VolatilePtr<Recorder>
{
public:
    class IGotDataCallback: public VolatilePtr<IGotDataCallback>
    {
    public:
        virtual ~IGotDataCallback() {}

        virtual void markNewlyRecordedData(Signal::Interval what)=0;
    };


    Recorder();
    virtual ~Recorder();

    virtual void startRecording() = 0;
    virtual void stopRecording() = 0;
    virtual bool isStopped() = 0;
    virtual bool canRecord() = 0;

    float time_since_last_update();
    void setDataCallback( IGotDataCallback::Ptr invalidator );
    Signal::Cache& data() { return _data; }

    // virtual from Signal::SourceBase
    // virtual std::string name() = 0;
    virtual float sample_rate() = 0;
    virtual unsigned num_channels() = 0;

    // overloaded from Signal::FinalSource
    virtual Signal::pBuffer read( const Signal::Interval& I );
    virtual Signal::IntervalType number_of_samples();
    virtual float length();

protected:
    QMutex _data_lock;
    Signal::Cache _data;
    IGotDataCallback::Ptr _invalidator;
    float _offset;
    boost::posix_time::ptime _start_recording, _last_update;
    Signal::IntervalType actual_number_of_samples();

    virtual float time();
};


} // namespace Adapters

#endif // RECORDER_H
