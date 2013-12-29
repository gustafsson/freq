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


} // namespace Tfr

#endif // TFRSTFTFILTER_H
