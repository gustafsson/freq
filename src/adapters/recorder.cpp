#include "recorder.h"
#include "sawe/configuration.h"

#include <QMutexLocker>

namespace Adapters {

Recorder::Recorder()
    :
    _data(0),
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


Signal::IntervalType Recorder::
        actual_number_of_samples()
{
    QMutexLocker lock(&_data_lock);
    Signal::IntervalType N = _data.number_of_samples();
    return N;
}



Signal::IntervalType Recorder::
        number_of_samples()
{
    return actual_number_of_samples();
/*    if (isStopped())
        return actual_number_of_samples();
    else
        return time() * sample_rate();*/
}

unsigned Recorder::
        num_channels()
{
    QMutexLocker lock(&_data_lock);
    if (Sawe::Configuration::mono())
        return _data.num_channels()?1:0;
    else
        return _data.num_channels();
}


Signal::pBuffer Recorder::
        read( const Signal::Interval& I )
{
    QMutexLocker lock(&_data_lock);
    // TODO why? return _data[channel].readFixedLength( I );
    return _data.read( I );
}



float Recorder::
        length()
{
    return isStopped() ? Signal::FinalSource::length() : time();
}


float Recorder::
        time()
{
    boost::posix_time::time_duration d = boost::posix_time::microsec_clock::local_time() - _start_recording;
    float dt = d.total_milliseconds()*0.001f;
    return dt + _offset;
}

} // namespace Adapters
