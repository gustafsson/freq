#ifndef CEPSTRUMFILTER_H
#define CEPSTRUMFILTER_H

#include "tfr/filter.h"

namespace Tfr {

class CepstrumKernelDesc: public Tfr::ChunkFilterDesc
{
public:
    CepstrumKernelDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter);

    Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* engine) const;

private:
    Tfr::pChunkFilter reentrant_cpu_chunk_filter_;
};


class CepstrumFilterDesc : public Tfr::TransformOperationDesc
{
public:
    CepstrumFilterDesc(Tfr::ChunkFilterDesc::Ptr filter_kernel_desc);
    CepstrumFilterDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter);

    // Must reimplement copy
    virtual OperationDesc::Ptr copy() const = 0;

    void transformDesc( Tfr::pTransformDesc m );
};


} // namespace Tfr


#endif // CEPSTRUMFILTER_H
