#ifndef TFRCWTFILTER_H
#define TFRCWTFILTER_H

#include "filter.h"

namespace Tfr {

class CwtFilter : public virtual Filter
{
public:
    CwtFilter( Signal::pOperation source=Signal::pOperation(),
               Tfr::pTransform transform=Tfr::pTransform() );

    virtual Filter::ChunkAndInverse readChunk( const Signal::Interval& I );


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

// TODO remove
class DummyCwtFilter: public CwtFilter {
public:
    virtual void operator()( Chunk& ) {}
};
} // namespace Tfr

#endif // TFRCWTFILTER_H
