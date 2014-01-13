#include "cepstrumfilter.h"
#include "cepstrum.h"
#include "signal/computingengine.h"

#include "neat_math.h"
#include <memory.h>

#define TIME_CepstrumFilter
//#define TIME_CepstrumFilter if(0)

using namespace Signal;

namespace Tfr {


CepstrumKernelDesc::
        CepstrumKernelDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter)
    :
      reentrant_cpu_chunk_filter_(reentrant_cpu_chunk_filter)
{
}


Tfr::pChunkFilter CepstrumKernelDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    if (dynamic_cast<Signal::ComputingCpu*>(engine))
        return reentrant_cpu_chunk_filter_;
    return Tfr::pChunkFilter();
}


CepstrumFilterDesc::
        CepstrumFilterDesc(Tfr::ChunkFilterDesc::Ptr filter_kernel_desc)
    :
      TransformOperationDesc(Tfr::pTransformDesc(), filter_kernel_desc)
{
    CepstrumDesc* desc;
    Tfr::pTransformDesc t(desc = new CepstrumDesc);
    desc->setWindow(Tfr::CepstrumDesc::WindowType_Hann, 0.75f);
    transformDesc( t );
}


CepstrumFilterDesc::
        CepstrumFilterDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter)
    :
      TransformOperationDesc(Tfr::pTransformDesc(), Tfr::ChunkFilterDesc::Ptr(new CepstrumKernelDesc(reentrant_cpu_chunk_filter)))
{
    CepstrumDesc* desc;
    Tfr::pTransformDesc t(desc = new CepstrumDesc);
    desc->setWindow(Tfr::CepstrumDesc::WindowType_Hann, 0.75f);
    transformDesc( t );
}


void CepstrumFilterDesc::
        transformDesc( Tfr::pTransformDesc m )
{
    const CepstrumDesc* desc = dynamic_cast<const CepstrumDesc*>(m.get ());

    EXCEPTION_ASSERT(desc);

    TransformOperationDesc::transformDesc (m);
}



} // namespace Signal
