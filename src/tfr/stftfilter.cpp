#include "stftfilter.h"
#include "stft.h"

#include "signal/computingengine.h"

#include "neat_math.h"
#include <memory.h>

//#define TIME_StftFilter
#define TIME_StftFilter if(0)

using namespace Signal;

namespace Tfr {

StftKernelDesc::
        StftKernelDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter)
    :
      reentrant_cpu_chunk_filter_(reentrant_cpu_chunk_filter)
{
}


Tfr::pChunkFilter StftKernelDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    if (dynamic_cast<Signal::ComputingCpu*>(engine))
        return reentrant_cpu_chunk_filter_;
    return Tfr::pChunkFilter();
}


StftFilterDesc::
        StftFilterDesc(Tfr::ChunkFilterDesc::Ptr filter_kernel_desc)
    :
      TransformOperationDesc(Tfr::pTransformDesc(), filter_kernel_desc)
{
    StftDesc* desc;
    Tfr::pTransformDesc t(desc = new StftDesc);
    desc->setWindow(Tfr::StftDesc::WindowType_Hann, 0.75f);
    transformDesc( t );
}


StftFilterDesc::
        StftFilterDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter)
    :
      TransformOperationDesc(Tfr::pTransformDesc(), Tfr::ChunkFilterDesc::Ptr(new StftKernelDesc(reentrant_cpu_chunk_filter)))
{
    StftDesc* desc;
    Tfr::pTransformDesc t(desc = new StftDesc);
    desc->setWindow(Tfr::StftDesc::WindowType_Hann, 0.75f);
    transformDesc( t );
}


void StftFilterDesc::
        transformDesc( Tfr::pTransformDesc m )
{
    const StftDesc* desc = dynamic_cast<const StftDesc*>(m.get ());

    EXCEPTION_ASSERT(desc);

    TransformOperationDesc::transformDesc (m);
}


} // namespace Signal
