#ifndef SIGNALCWTFILTER_H
#define SIGNALCWTFILTER_H

#include "tfr/filter.h"
#include "tfr/cwt.h"
#include "tfr/inversecwt.h"

namespace Tfr {

class CwtFilter : public Filter
{
public:
    CwtFilter( Signal::pSource source=Signal::pSource(),
               Tfr::pTransform transform=Tfr::pTransform() );

    virtual pChunk readChunk( const Signal::Interval& I );


    /**
      Get the Tfr::Transform for this operation.
      */
    Tfr::pTransform transform() const;

    /**
      Set the Tfr::Transform for this operation and update _invalid_samples.
      Will throw throw std::invalid_argument if 'm' is not an instance of
      Cwt.
      */
    void transform( Tfr::pTransform m );
};

} // namespace Tfr

#endif // SIGNALCWTFILTER_H
