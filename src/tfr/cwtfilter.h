#ifndef TFRCWTFILTER_H
#define TFRCWTFILTER_H

#pragma once

#include "filter.h"

namespace Tfr {

class CwtKernelDesc: public Tfr::FilterKernelDesc
{
public:
    CwtKernelDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter);

    Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* engine) const;

private:
    Tfr::pChunkFilter reentrant_cpu_chunk_filter_;
};


class CwtFilterDesc : public Tfr::FilterDesc
{
public:
    CwtFilterDesc(Tfr::FilterKernelDesc::Ptr filter_kernel_desc);
    CwtFilterDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter);

    void transformDesc( Tfr::pTransformDesc m );
};



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
    void applyFilter( ChunkAndInverse& chunk );

private:
    float   _previous_scales_per_octave;
    void    verify_scales_per_octave();
};


class DummyCwtFilter: public CwtFilter, public ChunkFilter::NoInverseTag {
public:
    void operator()( Chunk& ) {}
};
} // namespace Tfr

#endif // TFRCWTFILTER_H
