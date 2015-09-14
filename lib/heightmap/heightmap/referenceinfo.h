#ifndef HEIGHTMAP_REFERENCEINFO_H
#define HEIGHTMAP_REFERENCEINFO_H

#include "reference.h"
#include "position.h"
#include "blocklayout.h"
#include "visualizationparams.h"

#include "signal/intervals.h"

#include <boost/noncopyable.hpp>

namespace Heightmap {

class RegionFactory {
public:
    RegionFactory(const BlockLayout& block_size);

    /**
     * @brief getOverlapping produces the outer boundaries of all texels in the region, neighbouring references always overlap.
     * @param ref A reference to the region, it is interpreted in the coordinate system specified by block_size.
     * @return the region denotes the outer boundareis of the texels.
     * This is used as borders in an orthographic projection with the target texture as PBO.
     */
    Region getOverlapping(const Reference& ref) const;

    /**
     * @brief getVisibleRegion produces regions where neighbouring references are adjacent.
     * @param ref A reference to the region, it is interpreted in the coordinate system specified by block_size.
     * @return the region denotes the center of the texels.
     */
    Region getVisible(const Reference& ref) const;
private:
    const BlockLayout block_size_;
};


class ReferenceInfo {
public:
    enum BoundsCheck
    {
        BoundsCheck_HighS = 1,
        BoundsCheck_HighT = 2,
        //BoundsCheck_OutS = 4,
        //BoundsCheck_OutT = 8,
        BoundsCheck_All = 15
    };

    ReferenceInfo(const Reference&, const BlockLayout&, const VisualizationParams::const_ptr&);

    long double sample_rate() const;
    bool containsPoint(Position p) const;

    // returns false if the given BoundsCheck is out of bounds
    bool boundsCheck(BoundsCheck) const;

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
    float displayedTimeResolution(float ahz) const;
    float displayedFrequencyResolution(float hz1, float hz2 ) const;

    const BlockLayout block_layout_;
    const VisualizationParams::const_ptr visualization_params_;
    const Reference& reference_;
    const Region visible_region_;
};

} // namespace Heightmap

#endif // HEIGHTMAP_REFERENCEINFO_H
