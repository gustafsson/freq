#ifndef HEIGHTMAPREFERENCE_H
#define HEIGHTMAPREFERENCE_H

#include "blockconfiguration.h"
#include "position.h"
#include "signal/intervals.h"
#include "tfr/transform.h"

#include "tvector.h"
#include "deprecated.h"

namespace Heightmap {

class Reference {
public:
    tvector<2,int> log2_samples_size;
    tvector<2,unsigned> block_index;

    bool operator==(const Reference &b) const;
    // begin move out
    //DEPRECATED( bool containsPoint(Position p) const );
    bool containsPoint(Position p) const;
    enum BoundsCheck
    {
        BoundsCheck_HighS = 1,
        BoundsCheck_HighT = 2,
        BoundsCheck_OutS = 4,
        BoundsCheck_OutT = 8,
        BoundsCheck_All = 15
    };

    // returns false if the given BoundsCheck is out of bounds
    bool boundsCheck(BoundsCheck, const Tfr::TransformDesc* transform, float length) const;
    bool tooLarge(float length) const;
    std::string toString() const;

    /**
      Creates a SamplesIntervalDescriptor describing the entire range of the referenced block, including
      invalid samples.
      */
    Signal::Interval getInterval() const;
    Signal::Interval spannedElementsInterval(const Signal::Interval& I, Signal::Interval& spannedBlockSamples) const;
    // end moved out
    unsigned frequency_resolution() const;

    /** child references */
    Reference left() const;
    Reference right() const;
    Reference top() const;
    Reference bottom() const;

    /** sibblings, 3 other references who share the same parent */
    Reference sibbling1() const;
    Reference sibbling2() const;
    Reference sibbling3() const;

    /** sibblings */
    Reference sibblingLeft() const;
    Reference sibblingRight() const;
    Reference sibblingTop() const;
    Reference sibblingBottom() const;

    /** parent */
    Reference parent() const;
    Reference parentVertical() const;
    Reference parentHorizontal() const;

    //Reference( Collection* parent );
    Reference( const BlockConfiguration& block_config );
    ~Reference();

private:
    BlockConfiguration::Ptr block_config_;
};


} // namespace Heightmap

#endif // HEIGHTMAPREFERENCE_H
