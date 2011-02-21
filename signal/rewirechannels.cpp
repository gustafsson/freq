#include "rewirechannels.h"

namespace Signal {

const RewireChannels::SourceChannel RewireChannels::NOTHING = (unsigned)-1;

RewireChannels::
        RewireChannels(pOperation source)
            :
            Operation(source)
{
    resetMap();
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
    if (NOTHING != source_channel_)
        Operation::source()->set_channel( source_channel_ );
}


unsigned RewireChannels::
        get_channel()
{
    return output_channel_;
}


void RewireChannels::
        source(pOperation v)
{
    Operation::source(v);

    resetMap();
}


void RewireChannels::
        resetMap()
{
    scheme_.clear();

    if (Operation::source())
        num_channels( Operation::source()->num_channels() );
}


void RewireChannels::
        map(OutputChannel output_channel, SourceChannel source_channel)
{
    if ( output_channel >= num_channels() )
        num_channels( output_channel+1 );

    BOOST_ASSERT( source_channel < Operation::source()->num_channels() );

    scheme_[ output_channel ] = source_channel;
}


void RewireChannels::
        num_channels( unsigned N )
{
    unsigned M = scheme_.size();

    scheme_.resize( N );

    for (unsigned i=M; i<N; ++i)
        scheme_[i] = i;

    if (output_channel_ >= N && 0 < N)
        set_channel( N-1 );

    if (N != M)
        invalidate_samples( Signal::Interval(0, number_of_samples() ));
}

} // namespace Signal
