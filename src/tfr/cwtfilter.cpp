#include "cwtfilter.h"
#include "cwtchunk.h"
#include "cwt.h"
#include "neat_math.h"

#include <stringprintf.h>
#include <computationkernel.h>
#include <memory.h>
#include <demangle.h>

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


bool CwtFilter::
        applyFilter( ChunkAndInverse& chunkInv )
{
    Tfr::pChunk pchunk = chunkInv.chunk;

    TIME_CwtFilter TaskTimer tt("CwtFilter applying '%s' on chunk %s",
                                vartype(*this).c_str(),
                             pchunk->getInterval().toString().c_str());
    Tfr::CwtChunk* chunks = dynamic_cast<Tfr::CwtChunk*>( pchunk.get() );

    bool any = false;
    BOOST_FOREACH( const pChunk& chunk, chunks->chunks )
    {
        any |= (*this)( *chunk );
    }

    TIME_CwtFilter ComputationSynchronize();

    return any;
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
