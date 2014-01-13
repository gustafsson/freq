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


void CwtChunkFilter::
        operator()( ChunkAndInverse& c )
{
    Tfr::CwtChunk* cwtchunk = dynamic_cast<Tfr::CwtChunk*>(c.chunk.get ());
    BOOST_FOREACH(pChunk chunk, cwtchunk->chunks) {
        Tfr::ChunkAndInverse c2 = c;
        c2.chunk = chunk;

        subchunk(c2);
    }
}


CwtChunkFilterDesc::
        CwtChunkFilterDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter)
    :
      reentrant_cpu_chunk_filter_(reentrant_cpu_chunk_filter)
{
    EXCEPTION_ASSERT(dynamic_cast<CwtChunkFilter*>(reentrant_cpu_chunk_filter_.get ()));
}


Tfr::pChunkFilter CwtChunkFilterDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    if (dynamic_cast<Signal::ComputingCpu*>(engine))
        return reentrant_cpu_chunk_filter_;
    return Tfr::pChunkFilter();
}


CwtFilterDesc::
        CwtFilterDesc(Tfr::ChunkFilterDesc::Ptr filter_kernel_desc)
    :
      TransformOperationDesc(Tfr::pTransformDesc(), filter_kernel_desc)
{
    Cwt* desc;
    Tfr::pTransformDesc t(desc = new Cwt);
    transformDesc( t );
}


CwtFilterDesc::
        CwtFilterDesc(Tfr::pChunkFilter reentrant_cpu_chunk_filter)
    :
      TransformOperationDesc(
          Tfr::pTransformDesc(),
          reentrant_cpu_chunk_filter
            ? Tfr::ChunkFilterDesc::Ptr(new CwtChunkFilterDesc(reentrant_cpu_chunk_filter))
            : Tfr::ChunkFilterDesc::Ptr())
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

    TransformOperationDesc::transformDesc (m);
}


} // namespace Signal
