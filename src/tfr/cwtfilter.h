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


    virtual Signal::Interval requiredInterval (const Signal::Interval &I, pTransform t);

    Signal::Intervals include_time_support(Signal::Intervals);
    Signal::Intervals discard_time_support(Signal::Intervals);

    virtual void invalidate_samples(const Signal::Intervals& I);

protected:
    bool applyFilter( const ChunkAndInverse& chunk );

private:
    float   _previous_scales_per_octave;
    void    verify_scales_per_octave();
};


class DummyCwtFilter: public CwtFilter {
public:
    virtual bool operator()( Chunk& ) { return false; }
};
} // namespace Tfr

#endif // TFRCWTFILTER_H
