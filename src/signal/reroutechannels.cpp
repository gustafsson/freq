#include "reroutechannels.h"

namespace Signal {

const RerouteChannels::SourceChannel RerouteChannels::NOTHING = (unsigned)-1;

RerouteChannels::
        RerouteChannels(pOperation source)
            :
            Operation(source),
            output_channel_(0),
            source_channel_(0)
{
    resetMap();
}


pBuffer RerouteChannels::
        read( const Interval& I )
{
    if (NOTHING == source_channel_)
        return SourceBase::zeros( I );

    return Operation::read( I );
}


unsigned RerouteChannels::
        num_channels()
{
    return scheme_.size();
}


void RerouteChannels::
        set_channel(unsigned c)
{
    BOOST_ASSERT( c < num_channels() );

    output_channel_ = c;

    source_channel_ = scheme_[output_channel_];
    if (NOTHING != source_channel_)
        Operation::source()->set_channel( source_channel_ );
}


unsigned RerouteChannels::
        get_channel()
{
    return output_channel_;
}


void RerouteChannels::
        source(pOperation v)
{
    Operation::source(v);

    resetMap();
}


void RerouteChannels::
        invalidate_samples(const Intervals& I)
{
    unsigned N = Operation::num_channels();
    if (N != scheme_.size())
        num_channels( N );

    for (unsigned i=0; i<scheme_.size(); ++i)
    {
        if (scheme_[i] >= N && scheme_[i] != NOTHING)
            scheme_[i] = NOTHING;
    }

    Operation::invalidate_samples(I);
}


void RerouteChannels::
        resetMap()
{
    scheme_.clear();

    if (Operation::source())
        num_channels( Operation::source()->num_channels() );
}


void RerouteChannels::
        map(OutputChannel output_channel, SourceChannel source_channel)
{
    if ( output_channel >= num_channels() )
        num_channels( output_channel+1 );

    BOOST_ASSERT( source_channel < Operation::source()->num_channels() || NOTHING == source_channel);

    if (scheme_[ output_channel ] == source_channel)
        return;

    scheme_[ output_channel ] = source_channel;
    invalidate_samples( Signal::Intervals::Intervals_ALL );
}


void RerouteChannels::
        num_channels( unsigned N )
{
    unsigned M = scheme_.size();

    scheme_.resize( N );

    for (unsigned i=M; i<N; ++i)
        scheme_[i] = i;

    if (output_channel_ >= N && 0 < N)
        set_channel( N-1 );

    if (N != M)
        invalidate_samples( getInterval() );
}

} // namespace Signal
