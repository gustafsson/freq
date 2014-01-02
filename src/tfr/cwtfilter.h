#ifndef TFRCWTFILTER_H
#define TFRCWTFILTER_H

#pragma once

#include "filter.h"

namespace Tfr {

class CwtChunkFilter: public Tfr::ChunkFilter
{
    void operator()( ChunkAndInverse& chunk );

    virtual void subchunk( ChunkAndInverse& chunk ) = 0;
};


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

    // Must reimplement copy
    virtual OperationDesc::Ptr copy() const = 0;

    void transformDesc( Tfr::pTransformDesc m );
};

} // namespace Tfr

#endif // TFRCWTFILTER_H
