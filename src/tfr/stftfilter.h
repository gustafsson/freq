#ifndef TFRSTFTFILTER_H
#define TFRSTFTFILTER_H

#include "tfr/filter.h"

namespace Tfr {

class StftKernelDesc: public Tfr::FilterKernelDesc
{
public:
    StftKernelDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter);

    Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* engine) const;

private:
    Tfr::pChunkFilter reentrant_cpu_chunk_filter_;
};


class StftFilterDesc : public Tfr::FilterDesc
{
public:
    StftFilterDesc(Tfr::FilterKernelDesc::Ptr filter_kernel_desc);
    StftFilterDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter);

    void transformDesc( Tfr::pTransformDesc m );
};


class StftFilter : public Filter
{
public:
    StftFilter( Signal::pOperation source=Signal::pOperation(),
                Tfr::pTransform transform=Tfr::pTransform() );


    /**
      Computes the interval that computeChunk would need to work.
      */
    Signal::Interval requiredInterval( const Signal::Interval& I, Tfr::pTransform t );


    virtual void invalidate_samples(const Signal::Intervals& I);


    bool exclude_end_block;
};


} // namespace Tfr

#endif // TFRSTFTFILTER_H
