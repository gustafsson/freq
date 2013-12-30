#ifndef REROUTECHANNELS_H
#define REROUTECHANNELS_H

#include "operation.h"

#include <boost/noncopyable.hpp>

#include <vector>

namespace Signal {

/**
 * @brief The RerouteChannels class should rewire input channels into various output channels.
 */
class RerouteChannels : public Signal::OperationDesc
{
public:
    typedef unsigned SourceChannel;
    typedef unsigned OutputChannel;
    typedef std::vector<SourceChannel> MappingScheme;

    static const SourceChannel NOTHING;

    // OperationDesc
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    Signal::OperationDesc::Ptr copy() const;
    Signal::Operation::Ptr createOperation(Signal::ComputingEngine* engine=0) const;
    Extent extent() const;


    /**
      Creates a default mapping with no rewiring.
      */
    void resetMap();


    const MappingScheme& scheme() const { return scheme_; }


    /**
     * @brief map
     * @param output_channel the largest output_channel will define the number
     * of channels returned from Operation::process()
     * @param source_channel use NOTHING to create silence.
     * If the source channels doesn't exist the result will be silence.
     */
    void map(OutputChannel output_channel, SourceChannel source_channel);

private:
    MappingScheme scheme_;

public:
    static void test();
};

} // namespace Signal

#endif // REROUTECHANNELS_H
