#include "reroutechannels.h"

#include "signal/computingengine.h"

namespace Signal {

const RerouteChannels::SourceChannel RerouteChannels::NOTHING = (unsigned)-1;

class RerouteChannelsOperation : public Signal::Operation
{
public:
    RerouteChannelsOperation(RerouteChannels::MappingScheme scheme)
        :
        scheme_(scheme)
    {}


    pBuffer process( pBuffer b )
    {
        pBuffer r( new Buffer(b->sample_offset (), b->number_of_samples (), b->sample_rate (), (int)scheme_.size ()));
        for (unsigned i=0; i<scheme_.size (); ++i) {
            if (scheme_[i] < b->number_of_channels ())
                *r->getChannel (i) |= *b->getChannel (scheme_[i]);
        }
        return r;
    }


private:
    RerouteChannels::MappingScheme scheme_;
};


Signal::Interval RerouteChannels::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Signal::Interval RerouteChannels::
        affectedInterval( const Signal::Interval& I ) const
{
    return I;
}


Signal::OperationDesc::ptr RerouteChannels::
        copy() const
{
    RerouteChannels* c;
    Signal::OperationDesc::ptr o(c = new RerouteChannels);
    c->scheme_ = this->scheme_;
    return o;
}


Signal::Operation::ptr RerouteChannels::
        createOperation(Signal::ComputingEngine* engine) const
{
    if (engine == 0 || dynamic_cast<Signal::ComputingCpu*>(engine))
        return Signal::Operation::ptr(new RerouteChannelsOperation(scheme_));
    return Signal::Operation::ptr();
}


RerouteChannels::Extent RerouteChannels::
        extent() const
{
    RerouteChannels::Extent x;
    x.number_of_channels = scheme_.size ();
    return x;
}


void RerouteChannels::
        resetMap()
{
    scheme_.clear();
}


void RerouteChannels::
        map(OutputChannel output_channel, SourceChannel source_channel)
{
    if ( output_channel >= (int)scheme_.size() )
    {
        size_t M = scheme_.size();
        size_t N = output_channel+1;
        scheme_.resize( N );

        for (size_t i=M; i<N; ++i)
            scheme_[i] = NOTHING;
    }

    if (scheme_[ output_channel ] == source_channel)
        return;

    scheme_[ output_channel ] = source_channel;
}


} // namespace Signal


namespace Signal {

void RerouteChannels::
        test()
{
    // It should rewire input channels into various output channels.
    {
        EXCEPTION_ASSERTX(false, "not implemented");
    }
}
} // namespace Signal
