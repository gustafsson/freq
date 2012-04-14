#include "recorder.h"
#include "sawe/configuration.h"

#include <QMutexLocker>

namespace Adapters {

Recorder::Recorder()
    :
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


long unsigned Recorder::
        actual_number_of_samples()
{
    QMutexLocker lock(&_data_lock);
    long unsigned N = _data.number_of_samples();
    return N;
}



long unsigned Recorder::
        number_of_samples()
{
    if (isStopped())
        return actual_number_of_samples();
    else
        return time() * sample_rate();
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


void Recorder::
        set_channel(unsigned channel)
{
    QMutexLocker lock(&_data_lock);
    _data.set_channel(channel);
}


unsigned Recorder::
        get_channel()
{
    return _data.get_channel();
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
