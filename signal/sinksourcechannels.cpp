#include "sinksourcechannels.h"

namespace Signal {

SinkSourceChannels::
        SinkSourceChannels()
            :
            current_channel_(0)
{
    sinksources_.resize( 1 );
}


void SinkSourceChannels::
        put( pBuffer b )
{
    if (1>=b->channels())
        sinksources_[ current_channel_ ].put( b );

    else for (unsigned c=0; c < b->channels(); c++)
    {
        pBuffer q( new Buffer( b->getInterval(), b, c ));
        sinksources_[ c ].put( q );
    }
}


void SinkSourceChannels::
        putExpectedSamples( pBuffer b )
{
    sinksources_[ current_channel_ ].putExpectedSamples( b );
}


Intervals SinkSourceChannels::
        invalid_samples_current_channel()
{
    return sinksources_[ current_channel_ ].invalid_samples();
}


Intervals SinkSourceChannels::
        invalid_samples_all_channels()
{
    Intervals I;
    for (unsigned i=0; i<sinksources_.size(); ++i)
    {
        I |= sinksources_[ i ].invalid_samples();
    }
    return I;
}


void SinkSourceChannels::
        invalidate_samples(const Intervals& I)
{
    for (unsigned i=0; i<sinksources_.size(); ++i)
    {
        sinksources_[ i ].invalidate_samples( I );
    }
}


void SinkSourceChannels::
        invalidate_and_forget_samples(const Intervals& I)
{
    for (unsigned i=0; i<sinksources_.size(); ++i)
    {
        sinksources_[ i ].invalidate_and_forget_samples( I );
    }
}


void SinkSourceChannels::
        validate_samples_current_channels( const Intervals& I )
{
    sinksources_[ current_channel_ ].validate_samples( I );
}


void SinkSourceChannels::
        validate_samples_all_channels( const Intervals& I )
{
    for (unsigned i=0; i<sinksources_.size(); ++i)
    {
        sinksources_[ i ].validate_samples( I );
    }
}


void SinkSourceChannels::
        clear()
{
    for (unsigned i=0; i<sinksources_.size(); ++i)
    {
        sinksources_[ i ].clear();
    }
}


pBuffer SinkSourceChannels::
        read( const Interval& I )
{
    return sinksources_[ current_channel_ ].read( I );
}


pBuffer SinkSourceChannels::
        readAllChannelsFixedLength( const Interval& I )
{
    Signal::pBuffer b( new Signal::Buffer(I.first, I.count(), sample_rate(), num_channels() ));

    float* dst = b->waveform_data()->getCpuMemory();
    for (unsigned i=0; i<num_channels(); ++i)
    {
        Signal::pBuffer r = sinksources_[ i ].readFixedLength( I );
        float* src = r->waveform_data()->getCpuMemory();
        memcpy( dst + i*I.count(), src, I.count()*sizeof(float));
    }

    return b;
}


float SinkSourceChannels::
        sample_rate()
{
    return sinksources_[ current_channel_ ].sample_rate();
}


IntervalType SinkSourceChannels::
        number_of_samples()
{
    return samplesDesc().coveredInterval().count();
}


Interval SinkSourceChannels::
        getInterval()
{
    Intervals I;

    for (unsigned i=0; i<num_channels(); ++i)
    {
        I |= sinksources_[ i ].getInterval();
    }

    return I.coveredInterval();
}


unsigned SinkSourceChannels::
        num_channels()
{
    return sinksources_.size();
}


void SinkSourceChannels::
        set_channel(unsigned c)
{
    BOOST_ASSERT( c < num_channels());
    current_channel_ = c;
}


unsigned SinkSourceChannels::
        get_channel()
{
    return current_channel_;
}


SinkSource& SinkSourceChannels::
        channel(unsigned c)
{
    BOOST_ASSERT( c < num_channels());
    return sinksources_[c];
}


void SinkSourceChannels::
        setNumChannels(unsigned C)
{
    unsigned prevC = num_channels();
    if (C == prevC)
        return;

    Intervals I = samplesDesc() | invalid_samples();

    sinksources_.resize( C );
    for (unsigned i=prevC; i<C; ++i )
    {
        sinksources_[i].invalidate_samples( I );
    }
}


pBuffer SinkSourceChannels::
        first_buffer()
{
    return sinksources_[ current_channel_ ].first_buffer();
}


bool SinkSourceChannels::
        empty()
{
    return samplesDesc().empty();
}


Intervals SinkSourceChannels::
        samplesDesc_current_channel()
{
    return sinksources_[ current_channel_ ].samplesDesc();
}


Intervals SinkSourceChannels::
        samplesDesc_all_channels()
{
    Intervals I;
    for (unsigned i=0; i<sinksources_.size(); ++i)
    {
        I |= sinksources_[ i ].samplesDesc();
    }
    return I;
}


} // namespace Signal
