#ifndef REROUTECHANNELS_H
#define REROUTECHANNELS_H

#include "operation.h"

#include <boost/noncopyable.hpp>

#include <vector>

namespace Signal {

class RerouteChannels : public Signal::Operation, public boost::noncopyable
{
public:
    typedef unsigned SourceChannel;
    typedef unsigned OutputChannel;
    typedef std::vector<SourceChannel> MappingScheme;

    static const SourceChannel NOTHING;

    RerouteChannels(pOperation source);

    virtual pBuffer read( const Interval& I );
    virtual unsigned num_channels();
    virtual void set_channel(unsigned c);
    virtual unsigned get_channel();
    virtual void source(pOperation v);
    virtual pOperation source() { return Operation::source(); }

    /**
      Validate bindings.
      */
    virtual void invalidate_samples(const Intervals& I);


    /**
      Creates a default mapping with no rewiring.
      */
    void resetMap();


    const MappingScheme& scheme() const { return scheme_; }


    /**
      If output_channel is larger than num_channels(), num_channels will be
      increased to that number.

      It is an error to set source_channel bigger than source->num_channels(),
      except for the special value of setting source_channel to NOTHING to
      create silence.
      */
    void map(OutputChannel output_channel, SourceChannel source_channel);


    /**
      Set the number of output channels. If this value is larger than the
      currently mapped output_channels those channels will get a default
      value of no remapping, unless N is larger than source->num_channels()
      in which case they will be clamped to the last source channel.
      */
    void num_channels( unsigned N );

private:
    OutputChannel output_channel_;
    SourceChannel source_channel_;
    MappingScheme scheme_;
};

} // namespace Signal

#endif // REROUTECHANNELS_H
