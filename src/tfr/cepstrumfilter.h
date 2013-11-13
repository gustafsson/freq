#ifndef CEPSTRUMFILTER_H
#define CEPSTRUMFILTER_H

#include "tfr/filter.h"

namespace Tfr {

class CepstrumFilter : public Filter
{
public:
    CepstrumFilter( Signal::pOperation source=Signal::pOperation(),
                Tfr::pTransform transform=Tfr::pTransform() );


    /**
      This computes the Cepstrum chunk covering a given interval.
      */
    Signal::Interval requiredInterval( const Signal::Interval& I, Tfr::pTransform t );


    virtual void invalidate_samples(const Signal::Intervals& I);


    bool exclude_end_block;
};

} // namespace Tfr


#endif // CEPSTRUMFILTER_H
