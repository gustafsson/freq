#ifndef HEIGHTMAPREFERENCE_H
#define HEIGHTMAPREFERENCE_H

#include "position.h"
#include "signal/intervals.h"
#include "tfr/freqaxis.h"
#include <tvector.h>
#include <boost/shared_ptr.hpp>

#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED(func) func
#endif

namespace Heightmap {

class Collection;

class BlockConfiguration {
public:
    BlockConfiguration(Collection*);
    Collection* collection() const;
    void setCollection(Collection* c);
    unsigned samplesPerBlock() const;
    unsigned scalesPerBlock() const;
    float targetSampleRate() const;
    Tfr::FreqAxis display_scale() const;
    Tfr::FreqAxis transform_scale() const;
    float displayedTimeResolution(float ahz) const;
    float length() const;
private:
    Collection* collection_;
};


class Reference {
public:
    tvector<2,int> log2_samples_size;
    tvector<2,unsigned> block_index;

    bool operator==(const Reference &b) const;
    Region getRegion( unsigned samples_per_block, unsigned scales_per_block ) const;
    // begin move out
    DEPRECATED( Region getRegion() const );
    unsigned samplesPerBlock() const;
    unsigned scalesPerBlock() const;
    Collection* collection() const;
    void setCollection(Collection* c);

    long double sample_rate() const;

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

    Reference( Collection* parent );
    ~Reference();
private:
    boost::shared_ptr<BlockConfiguration> block_config_;
};


} // namespace Heightmap

#endif // HEIGHTMAPREFERENCE_H
