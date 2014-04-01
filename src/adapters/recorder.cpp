#include "recorder.h"
#include "sawe/configuration.h"

#include <QMutexLocker>

namespace Adapters {

Recorder::Recorder()
    :
    _data(),
    _offset(0)
{

}


Recorder::~Recorder()
{

}

float Recorder::
        time_since_last_update()
{
    if (isStopped())
        return 0;
    QMutexLocker lock(&_data_lock);
    boost::posix_time::time_duration d = boost::posix_time::microsec_clock::local_time() - _last_update;
    float dt = d.total_milliseconds()*0.001f;
    return std::min( dt, this->time() - _offset);
}


void Recorder::
        setDataCallback( IGotDataCallback::Ptr invalidator )
{
    _invalidator = invalidator;
}


Signal::IntervalType Recorder::
        actual_number_of_samples() const
{
    QMutexLocker lock(&_data_lock);
    Signal::IntervalType N = _data.spannedInterval().count();
    return N;
}



Signal::IntervalType Recorder::
        number_of_samples() const
{
    return actual_number_of_samples();
/*    if (isStopped())
        return actual_number_of_samples();
    else
        return time() * sample_rate();*/
}


Signal::pBuffer Recorder::
        read( const Signal::Interval& I )
{
    QMutexLocker lock(&_data_lock);
    if (_data.empty())
        return Signal::pBuffer(new Signal::Buffer(I, sample_rate(), num_channels ()));

    return _data.read( I );
}


float Recorder::
        length() const
{
    return isStopped() ? number_of_samples()/sample_rate() : time();
}


float Recorder::
        time() const
{
    boost::posix_time::time_duration d = boost::posix_time::microsec_clock::local_time() - _start_recording;
    float dt = d.total_milliseconds()*0.001f;
    return dt + _offset;
}

} // namespace Adapters
