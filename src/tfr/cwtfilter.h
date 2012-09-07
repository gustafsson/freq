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


    Signal::Intervals include_time_support(Signal::Intervals);
    Signal::Intervals discard_time_support(Signal::Intervals);

    virtual void invalidate_samples(const Signal::Intervals& I);

protected:
    void applyFilter( ChunkAndInverse& chunk );

private:
    float   _previous_scales_per_octave;
    void    verify_scales_per_octave();
};


class DummyCwtFilter: public CwtFilter {
public:
    virtual void operator()( Chunk& ) {}
};
} // namespace Tfr

#endif // TFRCWTFILTER_H
