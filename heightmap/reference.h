#ifndef HEIGHTMAPREFERENCE_H
#define HEIGHTMAPREFERENCE_H

#include "heightmap/position.h"
#include "signal/samplesintervaldescriptor.h"

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

    float sample_rate() const;
    float nFrequencies() const;

    bool containsSpectrogram() const;
    bool toLarge() const;

    /** child references */
    Reference left() const;
    Reference right() const;
    Reference top() const;
    Reference bottom() const;

    /** sibblings, 3 other references who share the same parent */
    Reference sibbling1() const;
    Reference sibbling2() const;
    Reference sibbling3() const;

    /** parent */
    Reference parent() const;

    /**
      Creates a SamplesIntervalDescriptor describing the entire range of the referenced block, including
      invalid samples.
      */
    Signal::SamplesIntervalDescriptor::Interval getInterval() const;
private:
    friend class Collection;

    Reference( Collection* parent );

    Collection* _collection;
};

} // namespace Heightmap

#endif // HEIGHTMAPREFERENCE_H
