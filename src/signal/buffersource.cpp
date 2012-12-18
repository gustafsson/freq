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


// OperationDesc
OperationDesc::Ptr BufferSource::
        copy() const
{
    return OperationDesc::Ptr(new BufferSource(buffer_));
}


Operation::Ptr BufferSource::
        createOperation(ComputingEngine*) const
{
    return Operation::Ptr(new BufferSourceOperation(buffer_));
}


// OperationSourceDesc
float BufferSource::
        getSampleRate() const
{
    return buffer_->sample_rate ();
}


float BufferSource::
        getNumberOfChannels() const
{
    return buffer_->number_of_channels ();
}


float BufferSource::
        getNumberOfSamples() const
{
    return buffer_->number_of_samples ();
}


void BufferSource::
        test()
{
    pBuffer b(new Buffer(Interval(60,70), 40, 7));
    for (unsigned c=0; c<b->number_of_channels (); ++c)
    {
        float *p = b->getChannel (c)->waveform_data ()->getCpuMemory ();
        for (int i=0; i<b->number_of_samples (); ++i)
            p[i] = c + i/(float)b->number_of_samples ();
    }
    BufferSource s(b);
    Operation::Ptr o = s.createOperation (0);
    Operation::test (o);

    pBuffer r(new Buffer(Interval(60,70), 40, 7));
    pBuffer d = o->process (r);
    EXCEPTION_ASSERT( *d == *b );

    r = pBuffer(new Buffer(Interval(61,71), 40, 7));
    d = o->process (r);
    for (unsigned c=0; c<b->number_of_channels (); ++c)
    {
        float *bp = b->getChannel (c)->waveform_data ()->getCpuMemory ();
        float *dp = d->getChannel (c)->waveform_data ()->getCpuMemory ();

        for (int i=0; i<b->number_of_samples ()-1; ++i)
            EXCEPTION_ASSERT(bp[1+i] == dp[i]);
        EXCEPTION_ASSERT(dp[9] == 0);
    }
}


} // namespace Signal
