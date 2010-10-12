#include "cwtfilter.h"
#include "cwtchunk.h"
#include "cwt.h"

#include <stringprintf.h>
#include <CudaException.h>
#include <memory.h>

#include <boost/foreach.hpp>

#define TIME_CwtFilter
//#define TIME_CwtFilter if(0)

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


Filter::ChunkAndInverse CwtFilter::
        readChunk( const Signal::Interval& I )
{
    unsigned firstSample = I.first, numberOfSamples = I.count();

    TIME_CwtFilter TaskTimer tt("CwtFilter::readChunk [%u, %u)", firstSample, numberOfSamples);
    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(transform().get());

    unsigned c = cwt.find_bin( cwt.nScales( sample_rate() ) - 1 );
    firstSample = firstSample>>c<<c;
    numberOfSamples = (numberOfSamples + (1<<c) - 1)>>c<<c;

    unsigned time_support = cwt.wavelet_time_support_samples( sample_rate() );

    // wavelet_std_samples gets stored in cwt so that inverse_cwt can take it
    // into account and create an inverse that is of the desired size.
    unsigned redundant_samples = 0;
    if (firstSample < time_support) redundant_samples = firstSample;
    else redundant_samples = time_support;

    unsigned first_valid_sample = firstSample;
    firstSample -= redundant_samples;

    unsigned smallest_ok_size = cwt.prev_good_size(0, sample_rate() );
    if (numberOfSamples<smallest_ok_size)
        numberOfSamples=smallest_ok_size;

    // These computations require a lot of memory allocations
    // If we encounter out of cuda memory, we decrease the required
    // memory in this while loop.
    while (true) //try
    {
        TIME_CwtFilter Intervals(I).print("CwtFilter subread");
        Filter::ChunkAndInverse ci;

        CwtFilter* f = dynamic_cast<CwtFilter*>(source().get());
        if ( f && f->transform() == transform()) {
            ci = f->readChunk( I );

        } else {
            unsigned L = redundant_samples + numberOfSamples + time_support;

            ci.inverse = _source->readFixedLength( Interval(firstSample,firstSample+ L) );

            // Compute the continous wavelet transform
            ci.chunk = (*transform())( ci.inverse );
        }

        // Only apply filter if it would affect these samples
        Intervals work(ci.chunk->getInterval());
        work -= affected_samples().inverse();

        if (work)
            ci.inverse.reset();

        // Apply filter
        if (work || !_try_shortcuts)
        {
            TIME_CwtFilter Intervals(ci.chunk->getInterval()).print("CwtFilter applying filter");
            Tfr::CwtChunk* chunks = dynamic_cast<Tfr::CwtChunk*>( ci.chunk.get() );

            BOOST_FOREACH( pChunk& chunk, chunks->chunks )
            //unsigned C = chunks->chunks.size();
            //pChunk chunk = chunks->chunks[C-2];
            {
                CudaException_CHECK_ERROR();
                (*this)( *chunk );
                CudaException_ThreadSynchronize();
                CudaException_CHECK_ERROR();
            }
        }

        TIME_CwtFilter Intervals(ci.chunk->getInterval()).print("CwtFilter after filter");

        return ci;
    }/* catch (const CufftException &x) {
        switch (x.getCufftError())
        {
            case CUFFT_EXEC_FAILED:
            case CUFFT_ALLOC_FAILED:
                break;
            default:
                throw;
        }

        unsigned newL = cwt.prev_good_size( numberOfSamples, sample_rate());
        if (newL < numberOfSamples ) {
            numberOfSamples = newL;

            TaskTimer("CwtFilter reducing chunk size to readRaw( %u, %u )\n%s", first_valid_sample, numberOfSamples, x.what() ).suppressTiming();
            continue;
        }

        // Try to decrease tf_resolution
        if (cwt.tf_resolution() > 1)
            cwt.tf_resolution(1/0.8f);

        if (cwt.tf_resolution() > exp(-2.f ))
        {
            cwt.tf_resolution( cwt.tf_resolution()*0.8f );
            float std_t = cwt.morlet_std_t(0, sample_rate());
            cwt.wavelet_std_t( std_t );

            TaskTimer("CwtFilter reducing tf_resolution to %g\n%s", cwt.tf_resolution(), x.what() ).suppressTiming();
            continue;
        }
        throw std::invalid_argument(printfstring("Not enough memory. Parameter 'wavelet_std_t=%g, tf_resolution=%g' yields a chunk size of %u MB.\n\n%s)",
                             cwt.wavelet_std_t(), cwt.tf_resolution(), cwt.wavelet_std_samples(sample_rate())*cwt.nScales(sample_rate())*sizeof(float)*2>>20, x.what()));
    } catch (const CudaException &x) {
        if (cudaErrorMemoryAllocation != x.getCudaError() )
            throw;

        unsigned newL = cwt.prev_good_size( numberOfSamples, sample_rate());
        if (newL < numberOfSamples ) {
            numberOfSamples = newL;

            TaskTimer("CwtFilter reducing chunk size to readRaw( %u, %u )\n%s", first_valid_sample, numberOfSamples, x.what() ).suppressTiming();
            continue;
        }

        // Try to decrease tf_resolution
        if (cwt.tf_resolution() > 1)
            cwt.tf_resolution(1/0.8f);

        if (cwt.tf_resolution() > exp(-2.f ))
        {
            cwt.tf_resolution( cwt.tf_resolution()*0.8f );
            float std_t = cwt.morlet_std_t(0, sample_rate());
            cwt.wavelet_std_t( std_t );

            TaskTimer("CwtFilter reducing tf_resolution to %g\n%s", cwt.tf_resolution(), x.what() ).suppressTiming();
            continue;
        }

        throw std::invalid_argument(printfstring("Not enough memory. Parameter 'wavelet_std_t=%g, tf_resolution=%g' yields a chunk size of %u MB.\n\n%s)",
                             cwt.wavelet_std_t(), cwt.tf_resolution(), cwt.wavelet_std_samples(sample_rate())*cwt.nScales(sample_rate())*sizeof(float)*2>>20, x.what()));
    }*/
}


Tfr::pTransform CwtFilter::
        transform() const
{
    return _transform ? _transform : Tfr::Cwt::SingletonP();
}


void CwtFilter::
        transform( Tfr::pTransform t )
{
    if (0 == dynamic_cast<Tfr::Cwt*>(t.get ()))
        throw std::invalid_argument("'transform' must be an instance of Tfr::Cwt");

    // even if '0 == t || transform() == t' the client
    // probably wants to reset everything when transform( t ) is called
    _invalid_samples = Intervals::Intervals_ALL;

    _transform = t;
}

} // namespace Signal
