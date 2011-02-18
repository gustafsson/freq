#include "rewirechannels.h"

namespace Signal {

const RewireChannels::SourceChannel RewireChannels::NOTHING = (unsigned)-1;

RewireChannels::
        RewireChannels(pOperation source)
            :
            Operation(source)
{
}


pBuffer RewireChannels::
        read( const Interval& I )
{
    if (NOTHING == source_channel_)
        return SourceBase::zeros( I );

    return Operation::read( I );
}


unsigned RewireChannels::
        num_channels()
{
    return scheme_.size();
}


void RewireChannels::
        set_channel(unsigned c)
{
    BOOST_ASSERT( c < num_channels() );

    output_channel_ = c;

    source_channel_ = scheme_[output_channel_];
}


unsigned RewireChannels::
        get_channel()
{
    return output_channel_;
}


void RewireChannels::
        resetMap()
{
    scheme_.clear();

    num_channels( source()->num_channels() );
}


void RewireChannels::
        map(OutputChannel output_channel, SourceChannel source_channel)
{
    if ( output_channel >= num_channels() )
        num_channels( output_channel+1 );

    BOOST_ASSERT( source_channel < source()->num_channels() );

    scheme_[ output_channel ] = source_channel;
}


void RewireChannels::
        num_channels( unsigned N )
{
    if (N < scheme_.size())
        scheme_.resize( N );
    else
    {
        unsigned M = scheme_.size();

        scheme_.resize( N );

        for (unsigned i=M; i<N; ++i)
            scheme_[i] = i;
    }
}

} // namespace Signal
