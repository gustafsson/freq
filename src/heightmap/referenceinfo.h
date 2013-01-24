#ifndef HEIGHTMAP_REFERENCEINFO_H
#define HEIGHTMAP_REFERENCEINFO_H

#include "reference.h"

#include "signal/intervals.h"

#include <boost/noncopyable.hpp>

namespace Heightmap {

class ReferenceInfo: boost::noncopyable {
public:
    ReferenceInfo(const BlockConfiguration*,const Reference&);

    Region getRegion() const;
    long double sample_rate() const;
    bool containsPoint(Position p) const;
    typedef Reference::BoundsCheck BoundsCheck;

    // returns false if the given BoundsCheck is out of bounds
    bool boundsCheck(BoundsCheck) const;
    bool tooLarge() const;
    std::string toString() const;

    /**
      Creates a SamplesIntervalDescriptor describing the entire range of the referenced block, including
      invalid samples.
      */
    Signal::Interval getInterval() const;
    Signal::Interval spannedElementsInterval(const Signal::Interval& I, Signal::Interval& spannedBlockSamples) const;

    static void test();

private:
    const BlockConfiguration* block_config_;
    const Reference& reference_;
};

} // namespace Heightmap

#endif // HEIGHTMAP_REFERENCEINFO_H
