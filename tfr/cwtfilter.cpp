#include "cwtfilter.h"
#include "cwtchunk.h"
#include "cwt.h"

#include <stringprintf.h>
#include <CudaException.h>
#include <memory.h>
#include <demangle.h>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#define TIME_CwtFilter
//#define TIME_CwtFilter if(0)

// #define DEBUG_CwtFilter
#define DEBUG_CwtFilter if(0)

using namespace Signal;

namespace Tfr {


CwtFilter::
        CwtFilter(pOperation source, Tfr::pTransform t)
:   Filter(source)
{
    if (!t)
        t = Tfr::Cwt::SingletonP();

    BOOST_ASSERT( dynamic_cast<Tfr::Cwt*>(t.get()));

    transform( t );
}


ChunkAndInverse CwtFilter::
        computeChunk( const Signal::Interval& I )
{
    Signal::IntervalType firstSample = I.first;

    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(transform().get());

    unsigned c = cwt.find_bin( cwt.nScales( sample_rate() ) - 1 );
    firstSample = firstSample>>c<<c;

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


    unsigned numberOfSamples = cwt.next_good_size( I.count()-1, sample_rate() );

    // hack to make it work without subsampling
    //unsigned numberOfSamples = cwt.next_good_size( 0, sample_rate() );

    unsigned L = redundant_samples + numberOfSamples + time_support;

    DEBUG_CwtFilter TaskTimer tt("L=%u, redundant=%u, num=%u, support=%u, first=%u",
                 L, redundant_samples, numberOfSamples, time_support, firstSample);

    ChunkAndInverse ci;

    ci.inverse = _source->readFixedLength( Interval(firstSample,
                                                    firstSample+L) );

    TIME_CwtFilter TaskTimer tt2("CwtFilter transforming %s for '%s'",
                                 ci.inverse->getInterval().toString().c_str(),
                                 vartype(*this).c_str());

    unsigned N_data=ci.inverse->number_of_samples();
    unsigned N_source=number_of_samples();
    if (firstSample<N_source)
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

    // Compute the continous wavelet transform
    ci.chunk = (*transform())( ci.inverse );

    return ci;
}


void CwtFilter::
        applyFilter( Tfr::pChunk pchunk )
{
    TIME_CwtFilter TaskTimer tt("CwtFilter applying '%s' on chunk %s",
                                vartype(*this).c_str(),
                             pchunk->getInterval().toString().c_str());
    Tfr::CwtChunk* chunks = dynamic_cast<Tfr::CwtChunk*>( pchunk.get() );

    foreach( pChunk& chunk, chunks->chunks )
    {
        (*this)( *chunk );
    }

    TIME_CwtFilter CudaException_ThreadSynchronize();
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

    // even if '0 == t || transform() == t' the client
    // probably wants to reset everything when transform( t ) is called
    _invalid_samples = Intervals::Intervals_ALL;

    _transform = t;
}

} // namespace Signal
