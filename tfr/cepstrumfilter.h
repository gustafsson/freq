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
    ChunkAndInverse computeChunk( const Signal::Interval& I );


    /**
      Get the Tfr::Transform for this operation.
      */
    Tfr::pTransform transform() const;


    /**
      Set the Tfr::Transform for this operation and update _invalid_samples.
      Will throw throw std::invalid_argument if 'm' is not an instance of
      Cepstrum.
      */
    void transform( Tfr::pTransform m );


    bool exclude_end_block;
};

} // namespace Tfr


#endif // CEPSTRUMFILTER_H
