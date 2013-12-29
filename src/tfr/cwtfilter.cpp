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


CwtFilter::
        CwtFilter(pOperation source, Tfr::pTransform t)
:   Filter(source),
    _previous_scales_per_octave(0)
{
    if (!t)
        t = pTransform(new Cwt());

    Cwt* c = dynamic_cast<Cwt*>(t.get());
    EXCEPTION_ASSERT( c );

    transform( t );
}


Interval CwtFilter::
        requiredInterval (const Interval &I, pTransform t)
{
    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(t.get());

    verify_scales_per_octave();

    unsigned chunk_alignment = cwt.chunk_alignment( sample_rate() );
    IntervalType firstSample = I.first;
    firstSample = align_down(firstSample, (IntervalType) chunk_alignment);

    unsigned time_support = cwt.wavelet_time_support_samples( sample_rate() );
    firstSample -= time_support;
    unsigned numberOfSamples = cwt.next_good_size( I.count()-1, sample_rate() );

    // hack to make it work without subsampling
#ifdef CWT_NOBINS
    numberOfSamples = cwt.next_good_size( 1, sample_rate() );
#endif

    unsigned L = time_support + numberOfSamples + time_support;

    return Interval(firstSample, firstSample+L);
}


void CwtFilter::
        applyFilter( ChunkAndInverse& chunkInv )
{
    Tfr::pChunk pchunk = chunkInv.chunk;

    TIME_CwtFilter TaskTimer tt("CwtFilter applying '%s' on chunk %s",
                                vartype(*this).c_str(),
                             pchunk->getInterval().toString().c_str());
    Tfr::CwtChunk* chunks = dynamic_cast<Tfr::CwtChunk*>( pchunk.get() );

    BOOST_FOREACH( const pChunk& chunk, chunks->chunks )
    {
        (*this)( *chunk );
    }

    TIME_CwtFilter ComputationSynchronize();
}


Intervals CwtFilter::
        include_time_support(Intervals I)
{
    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(transform().get());
    IntervalType n = cwt.wavelet_time_support_samples( sample_rate() );

    return I.enlarge( n );
}


Intervals CwtFilter::
        discard_time_support(Intervals I)
{
    Intervals r;
    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(transform().get());
    IntervalType n = cwt.wavelet_time_support_samples( sample_rate() );

    I.shrink( n );

    return r;
}


void CwtFilter::
        invalidate_samples(const Intervals& I)
{
    DeprecatedOperation::invalidate_samples( include_time_support(I) );
}

void CwtFilter::
        verify_scales_per_octave()
{
    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(transform().get());
    cwt.scales_per_octave( cwt.scales_per_octave(), sample_rate() );

    if (_previous_scales_per_octave != cwt.scales_per_octave())
    {
        bool first_verification = (0 == _previous_scales_per_octave);

        _previous_scales_per_octave = cwt.scales_per_octave();

        if (!first_verification)
            invalidate_samples(Intervals::Intervals_ALL);
    }
}


} // namespace Signal
