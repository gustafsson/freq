#include "buffersource.h"

#include "sawe/configuration.h"

#include <boost/foreach.hpp>

namespace Signal {


BufferSource::
        BufferSource( pMonoBuffer waveform )
{
    pBuffer b(new Buffer(waveform));
    setBuffer(b);
}


BufferSource::
        BufferSource( pBuffer buffer )
{
    setBuffer( buffer );
}


void BufferSource::
        setBuffer( pBuffer buffer )
{
    buffer_ = buffer;

    invalidate_samples (Intervals::Intervals_ALL);
}


pBuffer BufferSource::
        read( const Interval& I )
{
    Interval myInterval = buffer_->getInterval();
    Intervals i(I.first, I.first+1);
    if (i & myInterval)
    {
        return buffer_;
    }

    return zeros((Intervals(I) - myInterval).fetchFirstInterval());
}


float BufferSource::
        sample_rate()
{
    return buffer_->sample_rate();
}


void BufferSource::
        set_sample_rate( float fs )
{
    buffer_->set_sample_rate ( fs );

    invalidate_samples( Signal::Interval::Interval_ALL );
}


Signal::IntervalType BufferSource::
        number_of_samples()
{
    return buffer_->number_of_samples ();
}


unsigned BufferSource::
        num_channels()
{
    if (!buffer_)
        return 0;

    if (Sawe::Configuration::mono())
        return 1;
    else
        return buffer_->number_of_channels ();
}


} // namespace Signal
