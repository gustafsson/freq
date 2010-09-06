#ifndef SIGNALSTFTFILTER_H
#define SIGNALSTFTFILTER_H

#include "tfr/filter.h"
#include "tfr/cwt.h"
#include "tfr/inversecwt.h"

namespace Tfr {

class StftFilter : public Filter
{
public:
    StftFilter() {}
    StftFilter( Signal::pSource source );
    StftFilter( Signal::pSource source, Tfr::pTransform transform );

    virtual pChunk readChunk( const Signal::Interval& I );


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
};

} // namespace Tfr

#endif // SIGNALSTFTFILTER_H
