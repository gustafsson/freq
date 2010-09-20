#ifndef TFRSTFTFILTER_H
#define TFRSTFTFILTER_H

#include "tfr/filter.h"

namespace Tfr {

class StftFilter : public virtual Filter
{
public:
    StftFilter( Signal::pOperation source=Signal::pOperation(),
                Tfr::pTransform transform=Tfr::pTransform() );

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

#endif // TFRSTFTFILTER_H
