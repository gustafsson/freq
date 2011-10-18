#include "cwtfilter.h"
#include "cwtchunk.h"
#include "cwt.h"

#include <stringprintf.h>
#include <CudaException.h>
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
//    if (!t)
//        t = Tfr::Cwt::SingletonP();

    if (t)
    {
        BOOST_ASSERT( false );

        BOOST_ASSERT( dynamic_cast<Tfr::Cwt*>(t.get()));

        _transform = t;
    }
}


ChunkAndInverse CwtFilter::
        computeChunk( const Signal::Interval& I )
{
    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(transform().get());

    unsigned numberOfSamples = cwt.next_good_size( I.count()-1, sample_rate() );

    verify_scales_per_octave();
    // hack to make it work without subsampling
#ifdef CWT_NOBINS
    numberOfSamples = cwt.next_good_size( 1, sample_rate() );
#endif

    unsigned c = cwt.find_bin( cwt.nScales( sample_rate() ) - 1 );
    Signal::IntervalType firstSample = I.first;
    firstSample = firstSample/numberOfSamples*numberOfSamples;
    //firstSample = firstSample>>c<<c;
    BOOST_ASSERT( firstSample == firstSample>>c<<c );

    unsigned time_support = cwt.wavelet_time_support_samples( sample_rate() );

    // wavelet_std_samples gets stored in cwt so that inverse_cwt can take it
    // into account and create an inverse that is of the desired size.
    unsigned redundant_samples = time_support;
    if (firstSample < time_support)
    {
        redundant_samples = firstSample;
    }

    //unsigned first_valid_sample = firstSample;
    firstSample -= redundant_samples;

    unsigned L = redundant_samples + numberOfSamples + time_support;

    DEBUG_CwtFilter TaskTimer tt("L=%u, redundant=%u, num=%u, support=%u, first=%u",
                 L, redundant_samples, numberOfSamples, time_support, firstSample);

    ChunkAndInverse ci;

    {
        TIME_CwtFilterRead TaskTimer tt2("CwtFilter reading %s for '%s'",
                                         Interval(firstSample, firstSample+L).toString().c_str(),
                                     vartype(*this).c_str());

        ci.inverse = Operation::source()->readFixedLength( Interval(firstSample,
                                                        firstSample+L) );
    }

    TIME_CwtFilter TaskTimer tt2("CwtFilter transforming %s for '%s'",
                                 ci.inverse->getInterval().toString().c_str(),
                                 vartype(*this).c_str());

    unsigned N_data=ci.inverse->number_of_samples();
    unsigned N_source=number_of_samples();
    if(0) if (firstSample<N_source)
    {
        unsigned N=N_data;
        if (N_data>N_source-firstSample)
            N = N_source-firstSample;
        unsigned L=time_support/4;
        if (L>=N)
        {
            L=0;
            N=0;
        }

        float *p=ci.inverse->waveform_data()->getCpuMemory();
        for (unsigned i=0; i<L; ++i)
        {
            float k = i/(float)L;
            k = 1 - (1-k)*(1-k);

            p[i] *= k;
            p[N-1-i] *= k;
        }
        for (unsigned i=N;i<N_data;++i)
            p[i] = 0;
    }

    size_t free=0, total=0;
    DEBUG_CwtFilter cudaMemGetInfo(&free, &total);
    DEBUG_CwtFilter TaskInfo("free = %g, total = %g, inverse_size=%u, estimated = %g",
            free/1024./1024., total/1024./1024.,
            ci.inverse->number_of_samples(),
            cwt.required_gpu_bytes( numberOfSamples, sample_rate() )/1024./1024);

    // Compute the continous wavelet transform
    ci.chunk = (*transform())( ci.inverse );

    DEBUG_CwtFilter cudaMemGetInfo(&free, &total);
    DEBUG_CwtFilter TaskInfo("free = %g, total = %g",
             free/1024./1024., total/1024./1024. );

#ifdef _DEBUG
    BOOST_FOREACH( const pChunk& chunk, dynamic_cast<Tfr::CwtChunk*>(ci.chunk.get())->chunks )
    {
        Signal::Interval cii = chunk->getInterval();
        BOOST_ASSERT( cii & Signal::Intervals(I.first, I.first+1) );
    }
#endif

    return ci;
}


void CwtFilter::
        applyFilter( Tfr::pChunk pchunk )
{
    TIME_CwtFilter TaskTimer tt("CwtFilter applying '%s' on chunk %s",
                                vartype(*this).c_str(),
                             pchunk->getInterval().toString().c_str());
    Tfr::CwtChunk* chunks = dynamic_cast<Tfr::CwtChunk*>( pchunk.get() );

    BOOST_FOREACH( const pChunk& chunk, chunks->chunks )
    {
        (*this)( *chunk );
    }

    TIME_CwtFilter CudaException_ThreadSynchronize();
}


Signal::Intervals CwtFilter::
        include_time_support(Signal::Intervals I)
{
    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(transform().get());
    Signal::IntervalType n = cwt.wavelet_time_support_samples( sample_rate() );

    return I.enlarge( n );
}


Signal::Intervals CwtFilter::
        discard_time_support(Signal::Intervals I)
{
    Signal::Intervals r;
    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(transform().get());
    Signal::IntervalType n = cwt.wavelet_time_support_samples( sample_rate() );

    I.shrink( n );

    return r;
}


Tfr::pTransform CwtFilter::
        transform() const
{
    return _transform ? _transform : Tfr::Cwt::SingletonP();
}


void CwtFilter::
        transform( Tfr::pTransform t )
{
    if (0 == dynamic_cast<Tfr::Cwt*>( t.get()) )
        throw std::invalid_argument("'transform' must be an instance of Tfr::Cwt");

    if ( t == transform() && !_transform )
        t.reset();

    if (_transform == t )
        return;

    invalidate_samples( getInterval() );

    _transform = t;
}


void CwtFilter::
        invalidate_samples(const Intervals& I)
{
    Operation::invalidate_samples( include_time_support(I) );
}

void CwtFilter::
        verify_scales_per_octave()
{
    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(transform().get());

    if (_previous_scales_per_octave != cwt.scales_per_octave())
    {
        _previous_scales_per_octave = cwt.scales_per_octave();
        invalidate_samples(Signal::Intervals::Intervals_ALL);
    }
}


} // namespace Signal
