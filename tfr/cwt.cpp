#include "cwt.h"
#include "cwtchunk.h"
#include "stft.h"
#include "waveletkernel.h"
#include "supersample.h"

#include "signal/buffersource.h"

// gpumisc
//#include <cufft.h>
#include <throwInvalidArgument.h>
#include <computationkernel.h>
#include <neat_math.h>
#include <Statistics.h>

#ifdef USE_CUDA
#include "cudaglobalstorage.h"
#include "cudaMemsetFix.cu.h"
#endif

// std
#include <cmath>
#include <float.h>

// boost
#include <boost/lambda/lambda.hpp>
#include <boost/foreach.hpp>

#ifdef _MSC_VER
#include <msc_stdc.h>
#endif

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

//#define TIME_CWT if(0)
#define TIME_CWT

#define STAT_CWT if(0)
//#define STAT_CWT

#define TIME_CWTPART if(0)
//#define TIME_CWTPART

#define TIME_ICWT if(0)
//#define TIME_ICWT

#define DEBUG_CWT if(0)
//#define DEBUG_CWT

//#define CWT_NOBINS // Also change cwtfilter.cpp

//#define CWT_DISCARD_PREVIOUS_FT
#define CWT_DISCARD_PREVIOUS_FT if(0)

const bool AdjustToBin0 = true;

using namespace boost::lambda;

namespace Tfr {

pTransform Cwt::static_singleton;

Cwt::
        Cwt( float scales_per_octave, float wavelet_time_suppport )
#ifdef USE_CUDA
:   _min_hz( 20 ),
#else
:   _min_hz( 80 ), // the CPU version is so much slower, so ease it up a bit as default
#endif
    _scales_per_octave( 0 ),
    _tf_resolution( 2.5 ), // 2.5 is Ulfs magic constant
    _least_meaningful_fraction_of_r( 0.01f ),
    _least_meaningful_samples_per_chunk( 1024 ),
    _wavelet_time_suppport( wavelet_time_suppport ),
    _wavelet_def_time_suppport( wavelet_time_suppport ),
    _wavelet_scale_suppport( 6 ),
    _jibberish_normalization( 1 )
{
#ifdef USE_CUDA
    storageCudaMemsetFix = &cudaMemsetFix;
#endif
    this->scales_per_octave( scales_per_octave );
}


// static
Cwt& Cwt::
        Singleton()
{
    return *dynamic_cast<Cwt*>(SingletonP().get());
}


// static
pTransform Cwt::
        SingletonP()
{
    if (!static_singleton)
        static_singleton.reset( new Cwt() );
    return static_singleton;
}


pChunk Cwt::
        operator()( Signal::pBuffer buffer )
{    
#ifdef USE_CUDA
    try {
#endif

    boost::scoped_ptr<TaskTimer> tt;
    TIME_CWT tt.reset( new TaskTimer (
            "Forward CWT( buffer interval=%s, [%g, %g)%g# s)",
            buffer->getInterval().toString().c_str(),
            buffer->start(), buffer->length()+buffer->start(), buffer->length() ));

    TIME_CWT STAT_CWT Statistics<float>(buffer->waveform_data());

#ifdef USE_CUDA
    // move data to Gpu to start with, this will make new buffers created from this
    // to also be allocated on the Gpu
    CudaGlobalStorage::ReadOnly<1>(buffer->waveform_data());
#endif

    Signal::BufferSource bs( buffer );

    unsigned long offset = buffer->sample_offset;
    unsigned std_samples = wavelet_time_support_samples( buffer->sample_rate );
    //unsigned std_samples0 = time_support_bin0( buffer->sample_rate );
    unsigned long first_valid_sample = std_samples;
    unsigned added_silence = 0;

    if (0!=offset)
    {
        BOOST_ASSERT(buffer->number_of_samples() > 2*std_samples);
    }
    else
    {
        // Take care to do a proper calculation without requiring a larger fft
        // than would have been required if the same buffer size was used with
        // offset != 0
        BOOST_ASSERT(buffer->number_of_samples() > std_samples);

        unsigned L;
        if (buffer->number_of_samples() > 2*std_samples)
        {
            unsigned valid_samples = buffer->number_of_samples() - 2*std_samples;
            L = next_good_size( valid_samples - 1, buffer->sample_rate );
        }
        else
        {
            // compute smallest possible chunk, this will violate the above and
            // require a larger fft than would have been required if the same
            // buffer size was used with offset != 0. We'll take the smallest
            // possible that still makes any sense though.
            L = next_good_size( 0, buffer->sample_rate );
        }

        added_silence = L + 2*std_samples - buffer->number_of_samples();

        if (std_samples < added_silence)
            first_valid_sample = 0;
        else
            first_valid_sample = std_samples - added_silence;
    }

    // Align first_valid_sample with chunks (round upwards)
    first_valid_sample = align_up(offset + first_valid_sample, chunk_alignment(buffer->sample_rate)) - offset;

    BOOST_ASSERT( std_samples + first_valid_sample < buffer->number_of_samples());

    unsigned valid_samples = buffer->number_of_samples() - std_samples - first_valid_sample;
    // Align valid_samples with chunks (round downwards)
    unsigned alignment = chunk_alignment(buffer->sample_rate);
    valid_samples = align_down(valid_samples, alignment);
    BOOST_ASSERT( 0 < valid_samples );

    //unsigned L = 2*std_samples0 + valid_samples;
    //bool ispowerof2 = spo2g(L-1) == lpo2s(L+1);

    bool trypowerof2;
    {
        size_t free = availableMemoryForSingleAllocation();
        unsigned r, T0 = required_length( 0, buffer->sample_rate, r );
        unsigned smallest_L2 = spo2g(T0-1) - 2*r;
        size_t smallest_required2 = required_gpu_bytes(smallest_L2, buffer->sample_rate);
        trypowerof2 = smallest_required2 <= free;
    }

    DEBUG_CWT {
        size_t free = availableMemoryForSingleAllocation();
        TaskInfo("Free memory: %f MB. Required: %f MB",
             free/1024.f/1024.f,
             required_gpu_bytes(valid_samples, buffer->sample_rate )/1024.f/1024.f);
    }

    DEBUG_CWT TaskTimer("offset = %lu", offset).suppressTiming();
    DEBUG_CWT TaskTimer("std_samples = %lu", std_samples).suppressTiming();
    DEBUG_CWT TaskTimer("first_valid_sample = %lu", first_valid_sample).suppressTiming();
    DEBUG_CWT TaskTimer("valid_samples = %u", valid_samples).suppressTiming();
    DEBUG_CWT TaskTimer("added_silence = %u", added_silence).suppressTiming();

    // find all sub chunks
    unsigned prev_j = 0;
    unsigned n_j = nScales( buffer->sample_rate );

    pChunk ft;
    pChunk wt( new CwtChunk() );

    DEBUG_CWT
    {
        static bool list_scales = true;
        if (list_scales)
        {
            list_scales = false;

            TaskTimer tt("bins, scales per octave = %g, tf_resolution = %g, sigma = %g", _scales_per_octave, _tf_resolution, sigma());
            for (unsigned j = prev_j; j<n_j; ++j)
            {
                tt.getStream() << "j = " << j << ", "
                               << "hz = " << j_to_hz( buffer->sample_rate, j ) << ", "
                               << "bin = " << find_bin( j ) << ", "
                               << "time_support in samples = "
                               << wavelet_time_support_samples( buffer->sample_rate, j_to_hz( buffer->sample_rate, j )) << ", "
                               << "redundant in seconds = " << wavelet_time_support_samples( buffer->sample_rate, j_to_hz( buffer->sample_rate, j ))/buffer->sample_rate;
                tt.flushStream();
            }
        }
    }

    unsigned max_bin = find_bin( nScales( buffer->sample_rate ) - 1 );
    for( unsigned c=0; prev_j<n_j; ++c )
    {
        // find the biggest j that is required to be in this chunk
        unsigned next_j;
        if (c == max_bin)
            next_j = n_j;
        else
        {
            next_j = prev_j;

            for (unsigned j = prev_j; j<n_j; ++j)
            {
                if ( c == find_bin( j ) )
                    next_j = j;
            }

            if (next_j==prev_j)
                continue;
        }

        // If the biggest j required to be in this chunk is close to the end
        // 'n_j' then include all remaining scales in this chunk as well.
        if (2*(n_j - next_j) < n_j - prev_j || next_j+2>=n_j)
            next_j = n_j;

        // Move next_j forward one step so that it points to the first 'j'
        // that is not needed in this chunk part
        next_j = std::min(n_j, next_j+1);

        // Include next_j in this chunk so that parts can be interpolated
        // between in filters
        unsigned stop_j = std::min(n_j, next_j+1);
        unsigned nScales_value = nScales(buffer->sample_rate);
        BOOST_ASSERT( stop_j <= nScales_value);

        unsigned n_scales = stop_j - prev_j;
        float hz = j_to_hz(buffer->sample_rate, stop_j-1);
        DEBUG_CWT TaskTimer("c=%u, hz=%g, 2^c=%u, n_scales=%u", c, hz, 1<<c, n_scales).suppressTiming();

        unsigned sub_std_samples = wavelet_time_support_samples( buffer->sample_rate, hz );
        BOOST_ASSERT( sub_std_samples <= std_samples );

        unsigned sub_first_valid = sub_std_samples;
        unsigned sub_silence = 0;

        if (sub_first_valid > first_valid_sample)
        {
            sub_silence = sub_first_valid - first_valid_sample;
            sub_first_valid = first_valid_sample;
        }

        DEBUG_CWT TaskTimer("sub_std_samples=%u", sub_std_samples).suppressTiming();
        BOOST_ASSERT( sub_first_valid <= first_valid_sample );

        unsigned sub_start = offset + first_valid_sample - sub_first_valid;
        unsigned sub_length = sub_first_valid + valid_samples + sub_std_samples + sub_silence;
        BOOST_ASSERT( sub_length == valid_samples + 2*sub_std_samples );

        // Add some extra length to make length align with fast fft calculations
        unsigned extra = 0;
        if (trypowerof2)
            extra = spo2g(sub_length - 1) - sub_length;
        else
            extra = Fft::sChunkSizeG(sub_length - 1, chunkpart_alignment( c )) - sub_length;

        //this can be asserted if we compute valid interval based on the widest chunk
        if (!AdjustToBin0)
            if (nScales_value == stop_j && 0 < offset)
                BOOST_ASSERT( 0 == extra );

#ifdef _DEBUG
        unsigned sub_start_org = sub_start;
        unsigned sub_std_samples_org = sub_std_samples;
        unsigned sub_silence_org = sub_silence;
        unsigned sub_length_org = sub_length;
#endif

        sub_std_samples += extra/2;

        if (sub_start >= extra/2)
            sub_start -= extra/2;
        else
        {
            sub_silence += extra/2 - sub_start;
            sub_start = 0;
        }

        sub_length += extra;

        Signal::Interval subinterval(sub_start, sub_start + sub_length );

        CWT_DISCARD_PREVIOUS_FT
                ft.reset();

        if (!ft || ft->getInterval() != subinterval)
        {
            TIME_CWTPART TaskTimer tt(
                    "Computing forward fft on GPU of interval %s, was %s",
                    subinterval.toString().c_str(),
                    ft?ft->getInterval().toString().c_str():0 );

            Signal::pBuffer data;

            if (0<sub_silence)
            {
                //this can be asserted if we compute valid interval based on the widest chunk
                if (!AdjustToBin0)
                    BOOST_ASSERT(offset==0);
                TIME_CWTPART TaskTimer("Adding silence %u", sub_silence ).suppressTiming();
                Signal::Interval actualData = subinterval;
                actualData.last -= sub_silence;
                //this can be asserted if we compute valid interval based on the widest chunk
                if (!AdjustToBin0)
                    BOOST_ASSERT( (Signal::Interval(actualData.first, actualData.last) - buffer->getInterval()).empty() );
                Signal::BufferSource addSilence( bs.readFixedLength( actualData ) );
                data = addSilence.readFixedLength( subinterval );
            } else {
                //this can be asserted if we compute valid interval based on the widest chunk
                if (!AdjustToBin0)
                    BOOST_ASSERT( (Signal::Intervals(subinterval) - buffer->getInterval()).empty() );
                data = bs.readFixedLength( subinterval );
            }

            ComputationSynchronize();

            TIME_CWTPART TaskTimer t2("Doing fft");
            ft = Fft()( data );
            unsigned c = ft->getInterval().count();
            unsigned c2 = data->number_of_samples();
            BOOST_ASSERT( c == c2 );
            ComputationSynchronize();
        }

        // downsample the signal by shortening the fourier transform
        ((StftChunk*)ft.get())->setHalfs( c );
        pChunk chunkpart = computeChunkPart( ft, prev_j, n_scales );

        // The fft is most often bigger than strictly needed because it is
        // faster to compute lengths that are powers of 2.
        // However, to do proper merging we want to guarantee that all
        // chunkparts describe the exact same region. Thus we discard the extra
        // samples we added when padding to a power of 2
        chunkpart->first_valid_sample = (offset + first_valid_sample - subinterval.first) >> c;
        chunkpart->n_valid_samples = valid_samples >> c;

        DEBUG_CWT {
            TaskTimer tt("Intervals");
            tt.getStream() << " ft(c)=" << ft->getInterval().toString(); tt.flushStream();
            tt.getStream() << " adjusted chunkpart=" << chunkpart->getInversedInterval().toString(); tt.flushStream();
            tt.getStream() << " units=[" << (chunkpart->getInversedInterval().first >> (max_bin-c)) << ", "
                           << (chunkpart->getInversedInterval().last >> (max_bin-c)) << "), count="
                           << (chunkpart->getInversedInterval().count() >> (max_bin-c)); tt.flushStream();
        }

        ((CwtChunk*)wt.get())->chunks.push_back( chunkpart );

        // reset halfs if ft can be used for the next bin
        ((StftChunk*)ft.get())->setHalfs( 0 );

        prev_j = next_j;
    }


    wt->freqAxis = freqAxis( buffer->sample_rate );
    wt->chunk_offset = buffer->sample_offset + first_valid_sample;
    wt->first_valid_sample = 0;
    wt->n_valid_samples = valid_samples;
    wt->sample_rate = buffer->sample_rate;
    wt->original_sample_rate = buffer->sample_rate;

    DEBUG_CWT {
        size_t sum = 0;
        size_t alt = 0;
        BOOST_FOREACH( const pChunk& chunkpart, ((CwtChunk*)wt.get())->chunks )
        {
            size_t s = chunkpart->transform_data->getSizeInBytes1D();
            sum += s;
            size_t tmp_stft_and_fft = s + sizeof(Tfr::ChunkElement)*chunkpart->nSamples()*chunkpart->original_sample_rate/chunkpart->sample_rate;
            if (sum + tmp_stft_and_fft > alt)
                alt = sum + tmp_stft_and_fft; // biggest stft
        }

        if (alt>sum)
            sum = alt;

        size_t free = availableMemoryForSingleAllocation();
        TaskInfo("Free memory: %f MB. Allocated %f MB", free/1024.f/1024.f, sum/1024.f/1024.f);
    }

    DEBUG_CWT TaskTimer("wt->max_hz = %g, wt->min_hz = %g", wt->maxHz(), wt->minHz()).suppressTiming();

    TIME_CWT tt->getStream() << "Resulting interval = " << wt->getInterval().toString();
    TIME_CWT ComputationSynchronize();
    ComputationCheckError();

    TIME_CWT STAT_CWT
    {
        TaskInfo ti("stat cwt");
        BOOST_FOREACH( const pChunk& chunkpart, ((CwtChunk*)wt.get())->chunks )
        {
            DataStorageSize sz = chunkpart->transform_data->size();
            sz.width *= 2;
            Statistics<float> s( CpuMemoryStorage::BorrowPtr<float>( sz, (float*)CpuMemoryStorage::ReadOnly<2>(chunkpart->transform_data).ptr() ));
        }
    }

    return wt;
#ifdef USE_CUDA
    } catch (CufftException const& /*x*/) {
        TaskInfo("Cwt::operator() caught CufftException");
        throw;
    } catch (CudaException const& /*x*/) {
        TaskInfo("Cwt::operator() caught CudaException");
        throw;
    }
#endif
}


FreqAxis Cwt::
        freqAxis( float FS )
{
    FreqAxis fa;
    fa.setLogarithmic(
            get_min_hz( FS ),
            get_max_hz( FS ),
            nScales(FS) - 1 );
    return fa;
}


float Cwt::
        displayedTimeResolution( float FS, float hz )
{
    return morlet_sigma_samples(FS, hz) / FS;
}


Signal::Interval Cwt::
        validLength(Signal::pBuffer buffer)
{
    return Signal::Intervals(buffer->getInterval()).shrink( wavelet_time_support_samples(buffer->sample_rate) ).spannedInterval();
}


pChunk Cwt::
        computeChunkPart( pChunk ft, unsigned first_scale, unsigned n_scales )
{
    BOOST_ASSERT( n_scales > 1 || (first_scale == 0 && n_scales==1) );
    TIME_CWTPART TaskTimer tt("computeChunkPart first_scale=%u, n_scales=%u, (%g to %g Hz)",
                              first_scale, n_scales, j_to_hz(ft->original_sample_rate, first_scale+n_scales-1),
                              j_to_hz(ft->original_sample_rate, first_scale));

    pChunk intermediate_wt( new CwtChunkPart() );

    {
        DataStorageSize requiredWtSz( dynamic_cast<StftChunk*>(ft.get())->transformSize(), n_scales, 1 );
        TIME_CWTPART TaskTimer tt("Allocating chunk part (%u, %u, %u), %g kB",
                              requiredWtSz.width, requiredWtSz.height, requiredWtSz.depth,
                              requiredWtSz.width* requiredWtSz.height* requiredWtSz.depth * sizeof(Tfr::ChunkElement) / 1024.f);

        // allocate a new chunk
        intermediate_wt->transform_data.reset(new ChunkData( requiredWtSz ));

#ifdef USE_CUDA
        TIME_CWTPART {
            CudaGlobalStorage::useCudaPitch( intermediate_wt->transform_data, false );
            CudaGlobalStorage::ReadOnly<1>( ft->transform_data );
            CudaGlobalStorage::WriteAll<1>( intermediate_wt->transform_data );
        }
#endif

        TIME_CWTPART ComputationSynchronize();
    }

    unsigned half_sizes = dynamic_cast<Tfr::StftChunk*>(ft.get())->halfs();

    {
        TIME_CWTPART TaskTimer tt("inflating");

        // ft->sample_rate is related to intermediate_wt->sample_rate by
        // intermediate_wt->sample_rate == ft->n_valid_samples * ft->sample_rate
        // (except for numerical errors)
        intermediate_wt->sample_rate = ldexp(ft->original_sample_rate, -(int)half_sizes);
        intermediate_wt->original_sample_rate = ft->original_sample_rate;

        unsigned last_scale = first_scale + n_scales-1;
        intermediate_wt->freqAxis.setLogarithmic(
                get_max_hz(ft->original_sample_rate)*exp2f( last_scale/-_scales_per_octave ),
                get_max_hz(ft->original_sample_rate)*exp2f( first_scale/-_scales_per_octave ),
                intermediate_wt->nScales()-1 );

        DEBUG_CWT TaskInfo tinfo("scales [%u,%u]%u#, hz [%g, %g]",
                 first_scale, last_scale, n_scales,
                 intermediate_wt->maxHz(), intermediate_wt->minHz());

        DEBUG_CWT
        {
            TaskTimer("ft->sample_rate = %g", ft->sample_rate).suppressTiming();
            TaskTimer("ft->original_sample_rate = %g", ft->original_sample_rate).suppressTiming();
            TaskTimer("ft->halfs = %u", half_sizes).suppressTiming();
            TaskTimer("intermediate_wt->sample_rate = %g", intermediate_wt->sample_rate).suppressTiming();
            TaskTimer("intermediate_wt->min_hz = %g", intermediate_wt->minHz()).suppressTiming();
            TaskTimer("intermediate_wt->max_hz = %g", intermediate_wt->maxHz()).suppressTiming();
        }

        if( intermediate_wt->maxHz() > intermediate_wt->sample_rate/2 * (1.0+10*FLT_EPSILON) )
        {
            TaskInfo("intermediate_wt->max_hz = %g", intermediate_wt->maxHz());
            TaskInfo("intermediate_wt->sample_rate = %g", intermediate_wt->sample_rate);

            BOOST_ASSERT( intermediate_wt->maxHz() <= intermediate_wt->sample_rate/2 * (1.0+10*FLT_EPSILON) );
        }

        ::wtCompute( ft->transform_data,
                     intermediate_wt->transform_data,
                     intermediate_wt->sample_rate,
                     intermediate_wt->minHz(),
                     intermediate_wt->maxHz(),
                     1<<half_sizes,
                     _scales_per_octave, sigma(),
                     _jibberish_normalization );

        TIME_CWTPART ComputationSynchronize();
    }

    {
        // Compute the inverse fourier transform to get the filter banks back
        // in time space
        ChunkData::Ptr g = intermediate_wt->transform_data;
        DataStorageSize n = g->size();
        TIME_CWTPART TaskTimer tt("inverse stft(%u, %u)", n.width, n.height);

        intermediate_wt->chunk_offset = ft->getInterval().first;

        unsigned time_support = wavelet_time_support_samples( ft->original_sample_rate, intermediate_wt->minHz() );
        time_support >>= half_sizes;
        intermediate_wt->first_valid_sample = time_support;

        if (0==ft->chunk_offset)
            intermediate_wt->first_valid_sample=0;

        DEBUG_CWT {
            TaskTimer("time_support = %u", time_support).suppressTiming();
            TaskTimer("intermediate_wt->first_valid_sample=%u", intermediate_wt->first_valid_sample).suppressTiming();
            TaskTimer("ft->n_valid_samples=%u", ft->n_valid_samples).suppressTiming();
        }

        BOOST_ASSERT( time_support + intermediate_wt->first_valid_sample < ft->getInterval().count() );

        intermediate_wt->n_valid_samples = ft->getInterval().count() - time_support - intermediate_wt->first_valid_sample;

        Stft stft;
        stft.set_exact_chunk_size(n.width);
        stft.compute( g, g, Tfr::FftDirection_Inverse );

//        if (0 /* cpu version */ ) {
//            TIME_CWTPART TaskTimer tt("inverse ooura, redundant=%u+%u valid=%u",
//                                  intermediate_wt->first_valid_sample,
//                                  intermediate_wt->nSamples() - intermediate_wt->n_valid_samples - intermediate_wt->first_valid_sample,
//                                  intermediate_wt->n_valid_samples);

//            // Move to CPU
//            ChunkElement* p = g->getCpuMemory();

//            pChunk c( new CwtChunk );
//            for (unsigned h=0; h<n.height; h++) {
//                c->transform_data = CpuMemoryStorage::BorrowPtr<ChunkElement>(
//                        DataStorageSize(n.width, 1, 1), p + n.width*h );

//                DataStorage<float>::Ptr fb = Fft().backward( c )->waveform_data();
//                memcpy( p + n.width*h, fb->getCpuMemory(), fb->getSizeInBytes1D() );
//            }

//            // Move back to GPU
//            CudaGlobalStorage::ReadWrite<1>( g );
//        }
//        if (1 /* gpu version */ ) {
//            TIME_CWTPART TaskTimer tt("inverse cufft, redundant=%u+%u valid=%u, size(%u, %u)",
//                                  intermediate_wt->first_valid_sample,
//                                  intermediate_wt->nSamples() - intermediate_wt->n_valid_samples - intermediate_wt->first_valid_sample,
//                                  intermediate_wt->n_valid_samples,
//                                  n.width, n.height);

//            cufftComplex *d = (cufftComplex *)CudaGlobalStorage::ReadWrite<1>( g ).device_ptr();


//            {
//                //CufftHandleContext& fftctx = _fft_many[ n.width*n.height ];
//                CufftHandleContext fftctx; // TODO time optimization of keeping CufftHandleContext, "seems" unstable

//                {
//                    //TIME_CWTPART TaskTimer tt("Allocating inverse fft");
//                    fftctx(n.width, n.height);
//                }

//                DEBUG_CWT {
//                    size_t free = availableMemoryForSingleAllocation();
//                    size_t required = n.width*n.height*sizeof(float2)*2;
//                    TaskInfo("free = %g MB, required = %g MB", free/1024.f/1024.f, required/1024.f/1024.f);
//                }
//                TIME_CWTPART TaskInfo("n = { %u, %u, %u }, d = %ul", n.width, n.height, n.depth, g->numberOfElements());
//                CufftException_SAFE_CALL(cufftExecC2C(fftctx(n.width, n.height), d, d, CUFFT_INVERSE));
//            }

//            TIME_CWTPART ComputationSynchronize();
//        }
    }

    return intermediate_wt;
}


Signal::pBuffer Cwt::
        inverse( pChunk pchunk )
{
    ComputationCheckError();

    Tfr::CwtChunk* cwtchunk = dynamic_cast<Tfr::CwtChunk*>(pchunk.get());
    if (cwtchunk)
        return inverse(cwtchunk);

    Tfr::CwtChunkPart* cwtchunkpart = dynamic_cast<Tfr::CwtChunkPart*>(pchunk.get());
    if (cwtchunkpart)
        return inverse(cwtchunkpart);

    throw std::invalid_argument("Doesn't recognize chunk of type " + demangle( typeid(*pchunk.get())));
}


Signal::pBuffer Cwt::
        inverse( Tfr::CwtChunk* pchunk )
{
    boost::scoped_ptr<TaskTimer> tt;
    TIME_ICWT tt.reset( new TaskTimer("Inverse CWT( chunk %s, first_valid_sample=%u, [%g, %g] s)",
        pchunk->getInterval().toString().c_str(),
        pchunk->first_valid_sample,
        pchunk->startTime(),
        pchunk->endTime()
        ) );

    Signal::Interval v = pchunk->getInterval();
    Signal::pBuffer r( new Signal::Buffer( v.first, v.count(), pchunk->original_sample_rate ));
    memset( r->waveform_data()->getCpuMemory(), 0, r->waveform_data()->getSizeInBytes1D() );

    BOOST_FOREACH( pChunk& part, pchunk->chunks )
    {
        boost::scoped_ptr<TaskTimer> tt;
        DEBUG_CWT tt.reset( new TaskTimer("ChunkPart inverse, c=%g, [%g, %g] Hz",
            log2f(part->original_sample_rate/part->sample_rate),
            part->freqAxis.min_hz, part->freqAxis.max_hz()) );

        Signal::pBuffer inv = inverse(part);
        Signal::pBuffer super = SuperSample::supersample(inv, pchunk->sample_rate);

        DEBUG_CWT {
            tt->getStream()
                    << "Upsampled inv " << inv->getInterval().toString()
                    << " by factor " << pchunk->sample_rate/inv->sample_rate
                    << " to " << super->getInterval().toString(); tt->flushStream();

//            GpuCpuData<float> mdata( part->transform_data->getCpuMemory(),
//                                 make_cudaExtent( part->transform_data->getNumberOfElements1D(), 1, 1),
//                                 GpuCpuVoidData::CpuMemory, true );
        }

        //TaskInfo("super->getInterval() = %s, first_valid_sample = %u",
        //         super->getInterval().toString().c_str(), part->first_valid_sample);

        *r += *super;
    }

    BOOST_ASSERT( pchunk->getInterval() == r->getInterval() );
    TIME_ICWT {
        STAT_CWT Statistics<float>( r->waveform_data() );
    }

    TIME_ICWT ComputationSynchronize();

    return r;
}


Signal::pBuffer Cwt::
        inverse( Tfr::CwtChunkPart* pchunk )
{
    Chunk &chunk = *pchunk;

    DataStorageSize x = chunk.transform_data->size();

    Signal::pBuffer r( new Signal::Buffer(
            chunk.chunk_offset,
            x.width,
            chunk.sample_rate
            ));

    if (pchunk->original_sample_rate != pchunk->sample_rate)
    {
        // Skip last row
        x.height--;
    }

    ::wtInverse( chunk.transform_data,
                 r->waveform_data(),
                 x );

    TIME_ICWT ComputationSynchronize();

    return r;
}


float Cwt::
        wanted_min_hz() const
{
    return _min_hz;
}


float Cwt::
        get_min_hz( float fs ) const
{
    unsigned n_j = nScales( fs );
    return j_to_hz( fs, n_j - 1 );
}


void Cwt::
        set_wanted_min_hz(float value)
{
    if (value == _min_hz) return;

    _min_hz = value;
}


unsigned Cwt::
        nScales(float fs) const
{
    BOOST_ASSERT( _min_hz <= get_max_hz(fs) );
    float number_of_octaves = log2f(get_max_hz(fs)) - log2f(_min_hz);
    return 1 + ceil(number_of_octaves * scales_per_octave());
}


unsigned Cwt::
        nBins(float fs) const
{
    return find_bin( nScales(fs)-1 );
}


void Cwt::
        scales_per_octave( float value, float fs )
{
    scales_per_octave_internal( value );

    if (fs != 0)
    {
        // check available memory
        next_good_size(1, fs);
    }
}


void Cwt::
        scales_per_octave_internal( float value )
{
    if (value==_scales_per_octave) return;

    _scales_per_octave=value;

    float w = M_PI/2;
    float phi_sum = 0;
    float v = _scales_per_octave;
    float log2_a = 1.f / v;

    DEBUG_CWT TaskInfo ti("Cwt::scales_per_octave( %g )", value);
    for (int j=0; j<2*_scales_per_octave; ++j)
    {
        float aj = exp2f(log2_a * j );
        float q = (-w*aj + M_PI)*sigma();
        float phi_star = expf( -q*q );

        DEBUG_CWT TaskInfo("%d: %g", j, phi_star );
        phi_sum += phi_star;
    }

    float sigma_constant = sqrt( 4*M_PI*sigma() );
    _jibberish_normalization = 1 / (phi_sum * sigma_constant);

    DEBUG_CWT TaskInfo("phi_sum  = %g", phi_sum );
    DEBUG_CWT TaskInfo("sigma_constant  = %g", sigma_constant );
    DEBUG_CWT TaskInfo("_jibberish_normalization  = %g", _jibberish_normalization );
}


void Cwt::
        tf_resolution( float value )
{
    if (value == _tf_resolution) return;

    _tf_resolution = value;
}


float Cwt::
        sigma() const
{
    return _scales_per_octave/_tf_resolution;
}


float Cwt::
        morlet_sigma_samples( float fs, float hz ) const
{
    // float scale = hz/get_max_hz( fs );
    // float j = -_scales_per_octave*log2f(scale);
    // const float log2_a = 1.f / _scales_per_octave; // a = 2^(1/v), v = _scales_per_octave
    // float aj = exp2f( log2_a * j );

    float aj = get_max_hz( fs )/hz;

    return aj*sigma();
}


float Cwt::
        morlet_sigma_f( float hz ) const
{
    // float aj = get_max_hz( fs )/hz; // see morlet_sigma_t
    // float sigmaW0 = get_max_hz( fs ) / (2*M_PI*aj* sigma() );
    float sigmaW0 = hz / (2*M_PI* sigma() );
    return sigmaW0;
}


unsigned Cwt::
        wavelet_time_support_samples( float fs ) const
{
    return wavelet_time_support_samples( fs, get_min_hz(fs) );
}


unsigned Cwt::
        wavelet_time_support_samples( float fs, float hz ) const
{
    unsigned support_samples = std::ceil(morlet_sigma_samples( fs, hz ) * _wavelet_time_suppport);
    unsigned c = find_bin( hz_to_j( fs, hz ));
    // Make 2*support_samples align to chunk_alignment for the lowest frequency
    unsigned half_chunkpart_alignment = chunkpart_alignment( c )/2;
    support_samples = align_up( support_samples, half_chunkpart_alignment );
    return support_samples;
}


unsigned Cwt::
        required_length( unsigned current_valid_samples_per_chunk, float fs, unsigned &r )
{
    unsigned alignment = chunk_alignment( fs );
    if (AdjustToBin0)
        r = time_support_bin0( fs );
    else
    {
        r = wavelet_time_support_samples( fs );
        BOOST_ASSERT( ((2*r) % alignment) == 0 );
    }
    current_valid_samples_per_chunk = std::max((unsigned)(_least_meaningful_fraction_of_r*r), current_valid_samples_per_chunk);
    current_valid_samples_per_chunk = std::max(_least_meaningful_samples_per_chunk, current_valid_samples_per_chunk);
    current_valid_samples_per_chunk = align_up(current_valid_samples_per_chunk, alignment);
    unsigned T = r + current_valid_samples_per_chunk + r;
    if (AdjustToBin0)
        T = Fft::sChunkSizeG(T-1, chunkpart_alignment( 0 ));
    else
        T = Fft::sChunkSizeG(T-1, alignment);
    return T;
}


unsigned Cwt::
        next_good_size( unsigned current_valid_samples_per_chunk, float fs )
{
    DEBUG_CWT TaskInfo ti("next_good_size(%u, %g)", current_valid_samples_per_chunk, fs);
    unsigned r, T0 = required_length( 0, fs, r );

    size_t free = availableMemoryForSingleAllocation();

    // check power of 2 if possible, multi-radix otherwise
    unsigned smallest_L2 = spo2g(T0-1) - 2*r;
    size_t smallest_required2 = required_gpu_bytes(smallest_L2, fs);
    bool testPo2  = smallest_required2 <= free;

    unsigned T = required_length( current_valid_samples_per_chunk, fs, r );
    if (T - 2*r > current_valid_samples_per_chunk)
        T--;

    unsigned L;
    unsigned alignment = chunk_alignment( fs );

    if (AdjustToBin0)
    {
        unsigned nT;
        if (testPo2)
            nT = align_up( spo2g(T), chunkpart_alignment( 0 ));
        else
            nT = Fft::sChunkSizeG(T, chunkpart_alignment( 0 ));
        BOOST_ASSERT(nT != T);

        unsigned nL = nT - 2*r;
        L = align_down(nL, alignment);
        if (current_valid_samples_per_chunk >= L || alignment > L)
        {
            L = std::max((size_t)alignment, align_up(nL, alignment));
            T = L + 2*r;
            if (testPo2)
                nT = align_up( spo2g(T), chunkpart_alignment( 0 ));
            else
                nT = Fft::sChunkSizeG(T, chunkpart_alignment( 0 ));
            BOOST_ASSERT(nT != T);
            nL = nT - 2*r;
            L = align_down(nL, alignment);
        }
    }
    else
    {
        unsigned nT;
        if (testPo2)
            nT = align_up( spo2g(T), alignment);
        else
            nT = Fft::sChunkSizeG(T, alignment);
        BOOST_ASSERT(nT != T);

        L = nT - 2*r;
        BOOST_ASSERT( L % alignment == 0 );
    }

    size_t required = required_gpu_bytes(L, fs );

    if (free < required)
    {
        TaskInfo("next_good_size: current_valid_samples_per_chunk = %u (L=%u, r=%u) requires %f MB. Free: %f MB",
                 current_valid_samples_per_chunk, L, r,
                 required/1024.f/1024.f, free/1024.f/1024.f);
        return prev_good_size( L, fs );
    }
/*        unsigned nTtest = Fft::sChunkSizeG(T, chunk_alignment( fs ));
        unsigned Ltest = nTtest - 2*r;

        required = required_gpu_bytes(Ltest, fs );
        TaskInfo("next_good_size: Ltest = %u requires %f MB",
                 Ltest,
                 required/1024.f/1024.f);
        if (free < required)
            return prev_good_size( L, fs );
        else
            L = Ltest;
    }*/

    DEBUG_CWT TaskInfo("Cwt::next_good_size free = %g MB, required = %g MB", free/1024.f/1024.f, required/1024.f/1024.f);
    return L;
}


unsigned Cwt::
        prev_good_size( unsigned current_valid_samples_per_chunk, float fs )
{
    DEBUG_CWT TaskInfo ti("prev_good_size(%u, %g)", current_valid_samples_per_chunk, fs);

    // Check if the smallest possible size is ok memory-wise
    unsigned r, smallest_T = required_length( 1, fs, r );
    unsigned smallest_L = smallest_T - 2*r;
    unsigned alignment = chunk_alignment( fs );

    size_t free = availableMemoryForSingleAllocation();
    size_t smallest_required = required_gpu_bytes(smallest_L, fs);

    DEBUG_CWT TaskInfo(
            "prev_good_size: smallest_L = %u, chunk_alignment( %g ) = %u. Free: %f MB, required memory: %f MB",
             smallest_L, fs, alignment, free/1024.f/1024.f, smallest_required/1024.f/1024.f);

    BOOST_ASSERT( smallest_L + 2*r >= alignment );

    if (smallest_required <= free)
    {
        // current scales per octave is ok, take something smaller than
        // 'current_valid_samples_per_chunk'. Don't bother with
        // detailed sizes (sChunkSizeG), just use powers of 2 but make sure
        // it's still larger than chunk_alignment.

        // start searching from the requested number of valid samples per chunk
        unsigned T = required_length(current_valid_samples_per_chunk, fs, r);

        // check power of 2 if possible, multi-radix otherwise
        unsigned smallest_L2 = spo2g(smallest_L-1 + 2*r) - 2*r;
        size_t smallest_required2 = required_gpu_bytes(smallest_L2, fs);
        if (smallest_required2 <= free)
            smallest_L = smallest_L2;

        bool testPo2 = smallest_L == smallest_L2;

        while (true)
        {
            if (testPo2)
                T = align_up( lpo2s(T), 2);
            else
                T = Fft::lChunkSizeS(T, 2);

            //check whether we have reached the smallest acceptable length
            if ( T <= smallest_L + 2*r)
                return smallest_L;

            unsigned L = T - 2*r;
            L = std::max((size_t)alignment, align_down(L, alignment));

            size_t required = required_gpu_bytes(L, fs);

            if(required <= free)
                // found something that works, take it
                return L;
        }
    }

    DEBUG_CWT TaskInfo("prev_good_size: scales_per_octave was %g", scales_per_octave());

    // No, the current size doesn't fit in memory, find something smaller.
    largest_scales_per_octave( fs, scales_per_octave() );

    // scales per octave has been changed here, recompute 'r' and return
    // the smallest possible L
    smallest_T = required_length( 1, fs, r );
    smallest_L = smallest_T - 2*r;

    smallest_required = required_gpu_bytes(smallest_L, fs);
    DEBUG_CWT TaskInfo("prev_good_size: scales_per_octave is %g, smallest_L = %u, required = %f MB",
                       scales_per_octave(), smallest_L, smallest_required/1024.f/1024.f);

    return smallest_L;
}


void Cwt::
        largest_scales_per_octave(float fs, float scales, float last_ok )
{   
    if (scales_per_octave()<2)
    {
        scales = 0;
        last_ok = 2;
    }

    if (scales <= 0.1)
    {
        BOOST_ASSERT(last_ok != 0);
        scales_per_octave( last_ok );

        TaskInfo("largest_scales_per_octave is %g", scales_per_octave());
        return;
    }

    scales /= 2;

    if (is_small_enough( fs ))
    {
        last_ok = scales_per_octave();
        scales_per_octave(scales_per_octave() + scales);
    }
    else
    {
        scales_per_octave(scales_per_octave() - scales);
    }

    largest_scales_per_octave( fs, scales, last_ok );
}


bool Cwt::
        is_small_enough( float fs )
{
    unsigned r, T = required_length( 1, fs, r );
    unsigned L = T - 2*r;

    size_t free = availableMemoryForSingleAllocation();
    size_t required = required_gpu_bytes(L, fs );
    return free >= required;
}


size_t Cwt::
        required_gpu_bytes(unsigned valid_samples, float sample_rate) const
{
    DEBUG_CWT TaskInfo ti("required_gpu_bytes(%u, %g), scales_per_octave=%g, wavelet_time_suppport=%g", valid_samples, sample_rate, _scales_per_octave, _wavelet_time_suppport);

/*    unsigned r = wavelet_time_support_samples( sample_rate );
    unsigned max_bin = find_bin( nScales( sample_rate ) - 1 );
    long double sum = sizeof(float2)*(L + 2*r)*nScales( sample_rate )/(1+max_bin);

    // sum is now equal to the amount of memory required by the biggest CwtChunk
    // the stft algorithm needs to allocate the same amount of memory while working (+sum)
    // the rest of the chunks are in total basically equal to the size of the biggest chunk (+sum)
    sum = 3*sum;
*/
    size_t sum = 0;
    size_t alt = 0;

    unsigned max_bin = find_bin( nScales( sample_rate ) - 1 );
    unsigned
            prev_j = 0,
            n_j = nScales( sample_rate );

    unsigned std_samples = wavelet_time_support_samples( sample_rate );

    for( unsigned c=0; prev_j<n_j; ++c )
    {
        // find the biggest j that is required to be in this chunk
        unsigned next_j;
        if (c == max_bin)
            next_j = n_j;
        else
        {
            next_j = prev_j;

            for (unsigned j = prev_j; j<n_j; ++j)
            {
                if ( c == find_bin( j ) )
                    next_j = j;
            }

            if (next_j==prev_j)
                continue;
        }

        // If the biggest j required to be in this chunk is close to the end
        // 'n_j' then include all remaining scales in this chunk as well.
        if (2*(n_j - next_j) < n_j - prev_j || next_j+2>=n_j)
            next_j = n_j;

        // Move next_j forward one step so that it points to the first 'j'
        // that is not needed in this chunk part
        next_j = std::min(n_j, next_j+1);

        // Include next_j in this chunk so that parts can be interpolated
        // between in filters
        unsigned stop_j = std::min(n_j, next_j+1);

        unsigned n_scales = stop_j - prev_j;
        float hz = j_to_hz(sample_rate, stop_j-1);

        unsigned sub_std_samples = wavelet_time_support_samples( sample_rate, hz );
        unsigned sub_length = 2*sub_std_samples+valid_samples;

        unsigned L = valid_samples + 2*std_samples;
        bool ispowerof2 = spo2g(L-1) == lpo2s(L+1);
        if (ispowerof2)
            sub_length = spo2g(sub_length - 1);
        else
            sub_length = Fft::sChunkSizeG(sub_length - 1, chunk_alignment( sample_rate ));

        size_t s = 2*sizeof(Tfr::ChunkElement)*(sub_length >> c)*n_scales;

        sum += s;
        size_t tmp_stft_and_fft = s + 2*sizeof(Tfr::ChunkElement)*(sub_length);
        if (sum + tmp_stft_and_fft > alt)
            alt = sum + tmp_stft_and_fft;

        //TaskInfo("c = %d -> %f MB", c, (s+ tmp_stft_and_fft)/1024.f/1024.f);
        prev_j = next_j;
    }

    if (alt>sum)
        sum = alt;

    return sum < (size_t)-1 ? sum : (size_t)-1;
}


unsigned Cwt::
        chunk_alignment(float fs) const
{
    return chunkpart_alignment(nBins(fs));
}


unsigned Cwt::
        chunkpart_alignment(unsigned c) const
{
    unsigned sample_size = 1<<c;
    // In subchunk number 'c', one sample equals 'sample_size' samples in the
    // original sample rate.
    // To be able to use supersampling properly the length must be a multiple
    // of 2. Thus all subchunks need to be aligned to their respective
    // '2*sample_size'.
    return 2*sample_size;
}


unsigned Cwt::
        find_bin( unsigned j ) const
{
#ifdef CWT_NOBINS
    return 0;
#endif

    float v = _scales_per_octave;
    float log2_a = 1.f/v;
    float bin = log2_a * j - log2f( 1.f + _wavelet_scale_suppport/(2*M_PI*sigma()) );

    if (bin < 0)
        bin = 0;

    // could take maximum number of bins into account and meld all the last
    // ones into the same bin, effectively making the second last bin all empty
    // unsigned n_j = nScales( fs );

    return floor(bin);
}


unsigned Cwt::
        time_support_bin0( float fs ) const
{
    float lowesthz_inBin0;
    for (unsigned j=0; ;j++)
        if (0 != find_bin(j))
        {
            lowesthz_inBin0 = j_to_hz( fs, j-1 );
            break;
        }

    return wavelet_time_support_samples( fs, lowesthz_inBin0 );
}


void Cwt::
        resetSingleton()
{
    static_singleton.reset();
}


float Cwt::
        j_to_hz( float sample_rate, unsigned j ) const
{
    float v = _scales_per_octave;
    float hz = get_max_hz( sample_rate );
    hz *= exp2f(j/-v);
    return hz;
}


unsigned Cwt::
        hz_to_j( float sample_rate, float hz ) const
{
    float v = _scales_per_octave;
    float maxhz = get_max_hz( sample_rate );
    float j = -log2f(hz/maxhz)*v;
    j = floor(j+.5f);
    if (j<0)
        j=0;
    return (unsigned)j;
}



} // namespace Tfr
