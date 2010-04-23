#ifndef HEIGHTMAPREFERENCE_H
#define HEIGHTMAPREFERENCE_H

#include "heightmap-position.h"
#include "signal-samplesintervaldescriptor.h"

namespace Heightmap {

class Collection;

class Reference {
public:
    tvector<2,int> log2_samples_size;
    tvector<2,unsigned> block_index;

    bool operator==(const Reference &b) const;
    void getArea( Position &a, Position &b) const;
    unsigned sampleOffset() const;
    unsigned scaleOffset() const;
    unsigned samplesPerBlock() const;

    bool containsSpectrogram() const;
    bool toLarge() const;

    /** child references */
    Reference left();
    Reference right();
    Reference top();
    Reference bottom();

    /** sibblings, 3 other references who share the same parent */
    Reference sibbling1();
    Reference sibbling2();
    Reference sibbling3();

    /** parent */
    Reference parent();

    /**
      Creates a SamplesIntervalDescriptor describing the entire range of the referenced block, including
      invalid samples.
      */
    Signal::SamplesIntervalDescriptor::Interval getInterval();
private:
    friend class Collection;

    Reference( Collection* parent );

    Collection* _collection;
};

} // namespace Heightmap

#endif // HEIGHTMAPREFERENCE_H
