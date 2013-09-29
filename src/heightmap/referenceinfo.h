#ifndef HEIGHTMAP_REFERENCEINFO_H
#define HEIGHTMAP_REFERENCEINFO_H

#include "reference.h"
#include "tfrmapping.h"
#include "position.h"

#include "signal/intervals.h"
#include "tfr/transform.h"

#include <boost/noncopyable.hpp>

namespace Heightmap {

class ReferenceRegion {
public:
    ReferenceRegion(const BlockSize& block_size);

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

    ReferenceInfo(const Reference&, const TfrMapping&);
    ReferenceInfo(const Reference&, const BlockLayout&, const VisualizationParams::ConstPtr&);

    Region region() const;
    long double sample_rate() const;
    bool containsPoint(Position p) const;

    // returns false if the given BoundsCheck is out of bounds
    bool boundsCheck(BoundsCheck) const;

    /**
      Creates a SamplesIntervalDescriptor describing the entire range of the referenced block, including
      invalid samples.
      */
    Signal::Interval getInterval() const;
    Signal::Interval spannedElementsInterval(const Signal::Interval& I, Signal::Interval& spannedBlockSamples) const;

    Reference reference() const;


    std::string toString() const;

    template< class ostream_t > inline
    friend ostream_t& operator<<(ostream_t& os, const ReferenceInfo& r) {
        return os << r.toString();
    }

    static void test();

private:
    Tfr::FreqAxis transformScale() const;
    float displayedTimeResolution(float ahz) const;

    const BlockLayout block_layout_;
    const VisualizationParams::ConstPtr visualization_params_;
    const Reference& reference_;
    const Region r;
};

} // namespace Heightmap

#endif // HEIGHTMAP_REFERENCEINFO_H
