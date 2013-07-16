#include "buffersource.h"

#include "sawe/configuration.h"

#include "cpumemorystorage.h"

#include <boost/foreach.hpp>

namespace Signal {


Signal::pBuffer BufferSource::BufferSourceOperation::
        process(Signal::pBuffer b)
{
    Signal::pBuffer r(new Signal::Buffer(b->getInterval (), buffer_->sample_rate (), buffer_->number_of_channels ()));
    if (buffer_->getInterval ().contains (r->getInterval ()))
    {
        // Doesn't have to copy but can instead create a reference and use CopyOnWrite
    }
    else
    {
        // Need to clear data before merging with buffer. The easiest way to clear is to requires read access. Could also just zero the affect samples.
        // "should" allocate in the same memory as buffer uses.
        for (unsigned c=0; c<r->number_of_channels (); ++c)
            CpuMemoryStorage::ReadOnly<1>(r->getChannel (c)->waveform_data ());
    }
    *r |= *buffer_;
    return r;
}


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
Signal::Interval BufferSource::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Interval BufferSource::
        affectedInterval( const Interval& I ) const
{
    return I;
}


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


bool BufferSource::
        operator==(const OperationDesc& d) const
{
    if (!OperationDesc::operator == (d))
        return false;

    const BufferSource* b = dynamic_cast<const BufferSource*>(&d);
    if (b->buffer_ == buffer_)
        return true;
    if (*b->buffer_ == *buffer_)
        return true;
    return false;
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
    Operation::test (o, &s);

    pBuffer r(new Buffer(Interval(60,70), 40, 7));
    pBuffer d = o->process (r);
    EXCEPTION_ASSERT( *d == *b );

    r = pBuffer(new Buffer(Interval(61,71), 0, 1));
    d = o->process (r);
    EXCEPTION_ASSERT_EQUALS (b->number_of_channels (), d->number_of_channels ());
    EXCEPTION_ASSERT_EQUALS (b->number_of_samples (), d->number_of_samples ());
    EXCEPTION_ASSERT_EQUALS (r->getInterval (), d->getInterval ());
    for (unsigned c=0; c<b->number_of_channels (); ++c)
    {
        float *bp = b->getChannel (c)->waveform_data ()->getCpuMemory ();
        float *dp = d->getChannel (c)->waveform_data ()->getCpuMemory ();

        for (int i=0; i<b->number_of_samples ()-1; ++i)
            EXCEPTION_ASSERT_EQUALS (bp[1+i], dp[i]);
        EXCEPTION_ASSERT_EQUALS (dp[9], 0);
    }
}


} // namespace Signal
