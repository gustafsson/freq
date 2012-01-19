#ifndef TFRSTFTFILTER_H
#define TFRSTFTFILTER_H

#include "tfr/filter.h"

namespace Tfr {

class StftFilter : public Filter
{
public:
    StftFilter( Signal::pOperation source=Signal::pOperation(),
                Tfr::pTransform transform=Tfr::pTransform(),
                bool no_affected_samples=false);


    /**
      Computes the interval that computeChunk would need to work.
      */
    Signal::Interval requiredInterval( const Signal::Interval& I );


    /**
      This computes the Stft chunk covering a given interval.
      */
    ChunkAndInverse computeChunk( const Signal::Interval& I );


    /**
      Get the Tfr::Transform for this operation.
      */
    Tfr::pTransform transform() const;


    /**
      Set the Tfr::Transform for this operation and update _invalid_samples.
      Will throw throw std::invalid_argument if 'm' is not an instance of
      Stft.
      */
    void transform( Tfr::pTransform m );


    virtual void invalidate_samples(const Signal::Intervals& I);

    /// @overload Signal::Operation::affected_samples()
    virtual Signal::Intervals affected_samples()
    {
        if (no_affected_samples)
            return Signal::Intervals::Intervals();
        return Signal::Operation::affected_samples();
    }


    bool exclude_end_block;

private:
    /// @def false
    bool no_affected_samples;
};

} // namespace Tfr

#endif // TFRSTFTFILTER_H
