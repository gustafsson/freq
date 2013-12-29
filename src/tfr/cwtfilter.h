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

} // namespace Tfr

#endif // TFRCWTFILTER_H
