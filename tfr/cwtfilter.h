#ifndef TFRCWTFILTER_H
#define TFRCWTFILTER_H

#pragma once

#include "filter.h"

namespace Tfr {

class CwtFilter : public Filter
{
public:
    CwtFilter( Signal::pOperation source=Signal::pOperation(),
               Tfr::pTransform transform=Tfr::pTransform() );


    /**
      This computes the Cwt chunk covering a given interval.
      */
    ChunkAndInverse computeChunk( const Signal::Interval& I );


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

    Signal::Intervals include_time_support(Signal::Intervals);
    Signal::Intervals discard_time_support(Signal::Intervals);

    virtual void invalidate_samples(const Signal::Intervals& I);

protected:
    void applyFilter( Tfr::pChunk chunk );
};

// TODO remove
class DummyCwtFilter: public CwtFilter {
public:
    virtual void operator()( Chunk& ) {}
};
} // namespace Tfr

#endif // TFRCWTFILTER_H
