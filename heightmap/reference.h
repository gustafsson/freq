#ifndef HEIGHTMAPREFERENCE_H
#define HEIGHTMAPREFERENCE_H

#include "heightmap/position.h"
#include "signal/intervals.h"

#include <tvector.h>

namespace Heightmap {

class Collection;

class Reference {
public:
    tvector<2,int> log2_samples_size;
    tvector<2,unsigned> block_index;

    bool operator==(const Reference &b) const;
    void getArea( Position &a, Position &b) const;
    unsigned samplesPerBlock() const;
    unsigned scalesPerBlock() const;
    Collection* collection() const;
    void setCollection(Collection* c);

    float sample_rate() const;
    unsigned frequency_resolution() const;

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
    bool boundsCheck(BoundsCheck) const;
    bool tooLarge() const;
    std::string toString() const;

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

    /**
      Creates a SamplesIntervalDescriptor describing the entire range of the referenced block, including
      invalid samples.
      */
    Signal::Interval getInterval() const;
    Reference( Collection* parent );
private:
    friend class Collection;

    Collection* _collection;
};

} // namespace Heightmap

#endif // HEIGHTMAPREFERENCE_H
