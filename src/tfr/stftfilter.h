#ifndef TFRSTFTFILTER_H
#define TFRSTFTFILTER_H

#include "tfr/filter.h"

namespace Tfr {

class StftFilter : public Filter
{
public:
    StftFilter( Signal::pOperation source=Signal::pOperation(),
                Tfr::pTransform transform=Tfr::pTransform() );


    /**
      Computes the interval that computeChunk would need to work.
      */
    Signal::Interval requiredInterval( const Signal::Interval& I, Tfr::pTransform t );


    virtual void invalidate_samples(const Signal::Intervals& I);


    bool exclude_end_block;
};


} // namespace Tfr

#endif // TFRSTFTFILTER_H
