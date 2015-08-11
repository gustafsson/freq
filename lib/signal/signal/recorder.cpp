#include "recorder.h"
#include "log.h"

//#define LOG_DROPPED_FRAMES
#define LOG_DROPPED_FRAMES if(0)

namespace Signal {

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

    float dt = _last_update.elapsed ();
    return std::min( dt, this->time() - _offset);
}


void Recorder::
        setInvalidator( Signal::Processing::IInvalidator::ptr invalidator )
{
    _invalidator = invalidator;
}


Signal::IntervalType Recorder::
        actual_number_of_samples() const
{
    return _data->samples.spannedInterval().count();
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


float Recorder::
        sample_rate() const
{
    shared_state<Data> data = _data;
    if (data)
        return data.raw ()->sample_rate;
    return 0;
}


unsigned Recorder::
        num_channels() const
{
    shared_state<Data> data = _data;
    if (data)
        return data.raw ()->num_channels;
    return 0;
}


Signal::pBuffer Recorder::
        read( const Signal::Interval& I )
{
    if (_exception) {
        std::exception_ptr x = _exception;
        _exception = std::exception_ptr();
        std::rethrow_exception(x);
    }

    auto data = _data.read ();
    if (data->samples.empty())
        return Signal::pBuffer(new Signal::Buffer(I, data->sample_rate, data->num_channels));

    return data->samples.read( I );
}


float Recorder::
        length()
{
    if (isStopped ())
    {
        const auto data = _data;
        return data->samples.spannedInterval().last / data.raw ()->sample_rate;
    }
    else
        return time ();
}


float Recorder::
        time()
{
    const auto data = _data;
    float L = data->samples.spannedInterval().last / data.raw ()->sample_rate;
    float dt = _start_recording.elapsed ();
    float T = dt + _offset;
    if (T > L) {
        // dropped samples
        LOG_DROPPED_FRAMES Log ("recorder: dropped %d frames") % (int)((T-L) * data.raw ()->sample_rate + 0.5f);
        _offset -= T-L;
        T = L;
    }
    return T+0.1;
}

} // namespace Signal
