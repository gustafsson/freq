#ifndef HEIGHTMAP_REFERENCEINFO_H
#define HEIGHTMAP_REFERENCEINFO_H

#include "reference.h"
#include "blockconfiguration.h"
#include "position.h"

#include "signal/intervals.h"
#include "tfr/transform.h"

#include <boost/noncopyable.hpp>

namespace Heightmap {

class ReferenceRegion {
public:
    ReferenceRegion(const BlockSize& block_size);
    ReferenceRegion(const BlockConfiguration& block_size);

    Region operator()(const Reference& ref) const;

private:
    const BlockSize block_size_;
};


class ReferenceInfo {
public:
    enum BoundsCheck
    {
        BoundsCheck_HighS = 1,
        BoundsCheck_HighT = 2,
        BoundsCheck_OutS = 4,
        BoundsCheck_OutT = 8,
        BoundsCheck_All = 15
    };

    ReferenceInfo(const Reference&, const BlockConfiguration&);

    long double sample_rate() const;
    bool containsPoint(Position p) const;

    // returns false if the given BoundsCheck is out of bounds
    bool boundsCheck(BoundsCheck, const Tfr::TransformDesc*, float length) const;
    bool tooLarge(float length) const;

    /**
      Creates a SamplesIntervalDescriptor describing the entire range of the referenced block, including
      invalid samples.
      */
    Signal::Interval getInterval() const;
    Signal::Interval spannedElementsInterval(const Signal::Interval& I, Signal::Interval& spannedBlockSamples) const;

    Reference reference() const;
    BlockConfiguration block_config() const;


    std::string toString() const;

    template< class ostream_t > inline
    friend ostream_t& operator<<(ostream_t& os, const ReferenceInfo& r) {
        return os << r.toString();
    }

    static void test();
private:
    Tfr::FreqAxis transformScale(const Tfr::TransformDesc* transform) const;
    float displayedTimeResolution(float ahz, const Tfr::TransformDesc* transform) const;

    const BlockConfiguration& block_config_;
    const Reference& reference_;
};

} // namespace Heightmap

#endif // HEIGHTMAP_REFERENCEINFO_H
