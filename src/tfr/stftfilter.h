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
    Signal::Interval requiredInterval( const Signal::Interval& I, Tfr::pTransform t );


    /**
      This computes the Stft chunk covering a given interval.
      */
    ChunkAndInverse computeChunk( const Signal::Interval& I );


    virtual void invalidate_samples(const Signal::Intervals& I);

    /// @overload Signal::Operation::affected_samples()
    virtual Signal::Intervals affected_samples()
    {
        if (no_affected_samples)
            return Signal::Intervals();
        return Signal::Operation::affected_samples();
    }


    bool exclude_end_block;

private:
    /// @def false
    bool no_affected_samples;
};

} // namespace Tfr

#endif // TFRSTFTFILTER_H
