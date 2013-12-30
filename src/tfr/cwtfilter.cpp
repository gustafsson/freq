#include "cwtfilter.h"
#include "cwtchunk.h"
#include "cwt.h"
#include "neat_math.h"

#include "computationkernel.h"
#include <memory.h>
#include "demangle.h"
#include "signal/computingengine.h"

#include <boost/foreach.hpp>

//#define TIME_CwtFilter
#define TIME_CwtFilter if(0)

//#define TIME_CwtFilterRead
#define TIME_CwtFilterRead if(0)

// #define DEBUG_CwtFilter
#define DEBUG_CwtFilter if(0)

//#define CWT_NOBINS // Also change cwt.cpp

using namespace Signal;

namespace Tfr {

CwtKernelDesc::
        CwtKernelDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter)
    :
      reentrant_cpu_chunk_filter_(reentrant_cpu_chunk_filter)
{
}


Tfr::pChunkFilter CwtKernelDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    if (dynamic_cast<Signal::ComputingCpu*>(engine))
        return reentrant_cpu_chunk_filter_;
    return Tfr::pChunkFilter();
}


CwtFilterDesc::
        CwtFilterDesc(Tfr::FilterKernelDesc::Ptr filter_kernel_desc)
    :
      FilterDesc(Tfr::pTransformDesc(), filter_kernel_desc)
{
    Cwt* desc;
    Tfr::pTransformDesc t(desc = new Cwt);
    transformDesc( t );
}


CwtFilterDesc::
        CwtFilterDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter)
    :
      FilterDesc(Tfr::pTransformDesc(), Tfr::FilterKernelDesc::Ptr(new CwtKernelDesc(reentrant_cpu_chunk_filter)))
{
    Cwt* desc;
    Tfr::pTransformDesc t(desc = new Cwt);
    transformDesc( t );
}


void CwtFilterDesc::
        transformDesc( Tfr::pTransformDesc m )
{
    const Cwt* desc = dynamic_cast<const Cwt*>(m.get ());

    EXCEPTION_ASSERT(desc);

    FilterDesc::transformDesc (m);
}


} // namespace Signal
