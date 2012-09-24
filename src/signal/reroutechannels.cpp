#include "reroutechannels.h"

namespace Signal {

const RerouteChannels::SourceChannel RerouteChannels::NOTHING = (unsigned)-1;

RerouteChannels::
        RerouteChannels(pOperation source)
            :
            Operation(source)
{
    resetMap();
}


pBuffer RerouteChannels::
        read( const Interval& I )
{
    pBuffer b = Operation::read ( I );
    pBuffer r( new Buffer(b->sample_offset (), b->number_of_samples (), b->sample_rate (), scheme_.size ()));
    for (unsigned i=0; i<scheme_.size (); ++i)
        *r->getChannel (i) |= *b->getChannel (scheme_[i]);
    return r;
}


unsigned RerouteChannels::
        num_channels()
{
    return scheme_.size();
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
    bool invalidated = false;
    unsigned N = Operation::num_channels();
    if (N != scheme_.size())
    {
        invalidated = true;
        num_channels( N );
    }

    for (unsigned i=0; i<scheme_.size(); ++i)
    {
        if (scheme_[i] >= N && scheme_[i] != NOTHING)
            scheme_[i] = NOTHING;
    }

    if (!invalidated)
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
    bool invalidated = false;
    if ( output_channel >= num_channels() )
    {
        invalidated = true;
        num_channels( output_channel+1 );
    }

    BOOST_ASSERT( source_channel < Operation::source()->num_channels() || NOTHING == source_channel);

    if (scheme_[ output_channel ] == source_channel)
        return;

    scheme_[ output_channel ] = source_channel;

    if (!invalidated)
        invalidate_samples( Signal::Intervals::Intervals_ALL );
}


void RerouteChannels::
        num_channels( unsigned N )
{
    unsigned M = scheme_.size();

    scheme_.resize( N );

    for (unsigned i=M; i<N; ++i)
        scheme_[i] = i;

    if (N != M)
        invalidate_samples( getInterval() );
}

} // namespace Signal
