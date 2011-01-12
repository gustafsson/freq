#include "cwt.h"
#include "cwtchunk.h"
#include "stft.h"
#include "wavelet.cu.h"

#include "signal/buffersource.h"
#include "supersample.h"

#include <cufft.h>
#include <throwInvalidArgument.h>
#include <CudaException.h>
#include <neat_math.h>

#include <Statistics.h>

#include <cmath>
#include <boost/lambda/lambda.hpp>
#include <boost/foreach.hpp>

#ifdef _MSC_VER
#include <msc_stdc.h>
#endif

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#define TIME_CWT if(0)
//#define TIME_CWT

#define STAT_CWT if(0)
//#define STAT_CWT

#define TIME_CWTPART if(0)
//#define TIME_CWTPART

//#define TIME_ICWT if(0)
#define TIME_ICWT

#define DEBUG_CWT if(0)
//#define DEBUG_CWT

//#define CWT_NOBINS // Also change cwtfilter.cpp

//#define CWT_DISCARD_PREVIOUS_FT
#define CWT_DISCARD_PREVIOUS_FT if(0)

using namespace boost::lambda;

namespace Tfr {

std::map<unsigned, CufftHandleContext> Cwt::_fft_many;
pTransform Cwt::static_singleton;

Cwt::
        Cwt( float scales_per_octave, float wavelet_time_suppport, cudaStream_t stream )
:   _fft( /*stream*/ ),
    _stream( stream ),
    _min_hz( 20 ),
    _scales_per_octave( scales_per_octave ),
    _tf_resolution( 2.5 ), // 2.5 is Ulfs magic constant
    _wavelet_time_suppport( wavelet_time_suppport ),
    _wavelet_def_time_suppport( wavelet_time_suppport ),
    _wavelet_scale_suppport( 6 )
{
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
    try {
    boost::scoped_ptr<TaskTimer> tt;
    TIME_CWT tt.reset( new TaskTimer (
            "Forward CWT( buffer interval=%s, [%g, %g)%g# s)",
            buffer->getInterval().toString().c_str(),
            buffer->start(), buffer->length()+buffer->start(), buffer->length() ));

    TIME_CWT STAT_CWT Statistics<float>(buffer->waveform_data());

    Signal::BufferSource bs( buffer );

    unsigned long offset = buffer->sample_offset;
    unsigned std_samples = wavelet_time_support_samples( buffer->sample_rate );
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
            L = next_good_size( 0, buffer->sample_rate );
        }

        added_silence = L + 2*std_samples - buffer->number_of_samples();

        if (std_samples < added_silence)
            first_valid_sample = 0;
        else
            first_valid_sample = std_samples - added_silence;
    }

    // Align blocks with max_bin, max_bin is the bin of the last scale
    unsigned max_bin = find_bin( nScales( buffer->sample_rate ) - 1 );

    // Align first_valid_sample with max_bin (round upwards)
    first_valid_sample = ((offset + first_valid_sample + (1<<max_bin) - 1)>>max_bin<<max_bin) - offset;

    BOOST_ASSERT( std_samples + first_valid_sample < buffer->number_of_samples());

    unsigned valid_samples = buffer->number_of_samples() - std_samples - first_valid_sample;
    // Align valid_samples with max_bin (round downwards)
    valid_samples = valid_samples>>max_bin<<max_bin;
    BOOST_ASSERT( 0 < valid_samples );

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
        if (2*(n_j - next_j) < n_j - prev_j)
            next_j = n_j;

        // Move next_j forward one step so that it points to the first 'j'
        // that is not needed in this chunk part
        next_j = std::min(n_j, next_j+1);

        // Include next_j in this chunk so that parts can be interpolated
        // between in filters
        unsigned stop_j = std::min(n_j, next_j+1);

        unsigned n_scales = stop_j - prev_j;
        float hz = j_to_hz(buffer->sample_rate, stop_j-1);
        DEBUG_CWT TaskTimer("c=%u, hz=%g, 2^c=%u, n_scales=%u", c, hz, 1<<c, n_scales).suppressTiming();

        unsigned sub_std_samples = wavelet_time_support_samples( buffer->sample_rate, hz );
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

        // Add some extra length to make length a power of 2 for faster fft
        // calculations
        unsigned extra = spo2g(sub_length - 1) - sub_length;

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

        //CWT_DISCARD_PREVIOUS_FT
                ft.reset();

        if (!ft ||
            (Signal::Intervals)ft->getInterval() !=
            (Signal::Intervals)subinterval)
        {
            TIME_CWTPART TaskTimer tt(
                    "Computing forward fft on GPU of interval %s",
                    subinterval.toString().c_str());

            Signal::pBuffer data;

            if (0<sub_silence)
            {
                BOOST_ASSERT(offset==0);
                TIME_CWTPART TaskTimer("Adding silence %u", sub_silence ).suppressTiming();
                Signal::Interval actualData = subinterval;
                actualData.last -= sub_silence;
                BOOST_ASSERT( (Signal::Intervals(actualData) - buffer->getInterval()).empty() );

                Signal::BufferSource addSilence( bs.readFixedLength( actualData ) );
                data = addSilence.readFixedLength( subinterval );
            } else {
                BOOST_ASSERT( (Signal::Intervals(subinterval) - buffer->getInterval()).empty() );
                data = bs.readFixedLength( subinterval );
            }

            if ( 0 ) // Set S-curved window on both sides.
            {
                float* p = data->waveform_data()->getCpuMemory();

                unsigned long edge = subinterval.last - sub_silence;
                int ramp = sub_std_samples/4;
                for (int i = 0; i <= 2*ramp; ++i)
                {
                    float f = abs(i-ramp)/(float)ramp;
                    f = 3*f*f - 2*f*f*f;
                    p[ (edge - ramp + i)%sub_length ] *= f;
                }
            }

            ft = _fft( data );
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


    wt->axis_scale = AxisScale_Logarithmic;
    wt->chunk_offset = buffer->sample_offset + first_valid_sample;
    wt->first_valid_sample = 0;
    wt->max_hz = get_max_hz( buffer->sample_rate );
    wt->min_hz = get_min_hz( buffer->sample_rate );
    wt->n_valid_samples = valid_samples;
    wt->order = Chunk::Order_row_major;
    wt->sample_rate = buffer->sample_rate;
    wt->original_sample_rate = buffer->sample_rate;

    DEBUG_CWT TaskTimer("wt->max_hz = %g, wt->min_hz = %g", wt->max_hz, wt->min_hz).suppressTiming();

    TIME_CWT tt->getStream() << "Resulting interval = " << wt->getInterval().toString();
    TIME_CWT CudaException_ThreadSynchronize();

    return wt;

    } catch (CufftException const& /*x*/) {
        TaskInfo("Cwt::operator() caught CufftException, calling _fft_many.clear()");
        _fft_many.clear();
        throw;
    } catch (CudaException const& /*x*/) {
        TaskInfo("Cwt::operator() caught CudaException, calling _fft_many.clear()");
        _fft_many.clear();
        throw;
    }
}


pChunk Cwt::
        computeChunkPart( pChunk ft, unsigned first_scale, unsigned n_scales )
{
    TIME_CWTPART TaskTimer tt("computeChunkPart first_scale=%u, n_scales=%u, (%g to %g Hz)",
                              first_scale, n_scales, j_to_hz(ft->original_sample_rate, first_scale+n_scales-1),
                              j_to_hz(ft->original_sample_rate, first_scale));

    pChunk intermediate_wt( new CwtChunkPart() );

    {
        cudaExtent requiredWtSz = make_cudaExtent( ft->nScales(), n_scales, 1 );
        TIME_CWTPART TaskTimer tt("Allocating chunk part (%u, %u, %u), %g kB",
                              requiredWtSz.width, requiredWtSz.height, requiredWtSz.depth,
                              requiredWtSz.width* requiredWtSz.height* requiredWtSz.depth * sizeof(float2) / 1024.f);

        // allocate a new chunk
        intermediate_wt->transform_data.reset(new GpuCpuData<float2>( 0, requiredWtSz, GpuCpuVoidData::CudaGlobal ));

        TIME_CWTPART {
            ft->transform_data->getCudaGlobal();
            intermediate_wt->transform_data->getCudaGlobal();
        }

        TIME_CWTPART CudaException_ThreadSynchronize();
    }

    unsigned half_sizes;
    {
        Tfr::StftChunk* stchunk = dynamic_cast<Tfr::StftChunk*>(ft.get());
        half_sizes = stchunk->halfs();
    }

    {        
        TIME_CWTPART TaskTimer tt("inflating");

        // ft->sample_rate is related to intermediate_wt->sample_rate by
        // intermediate_wt->sample_rate == ft->n_valid_samples * ft->sample_rate
        // (except for numerical errors)
        intermediate_wt->sample_rate = ldexp(ft->original_sample_rate, -(int)half_sizes);
        intermediate_wt->original_sample_rate = ft->original_sample_rate;

        unsigned last_scale = first_scale + n_scales-1;
        intermediate_wt->min_hz = get_max_hz(ft->original_sample_rate)*exp2f( last_scale/-_scales_per_octave );
        intermediate_wt->max_hz = get_max_hz(ft->original_sample_rate)*exp2f( first_scale/-_scales_per_octave );

        DEBUG_CWT TaskInfo tinfo("scales [%u,%u]%u#, hz [%g, %g]",
                 first_scale, last_scale, n_scales,
                 intermediate_wt->max_hz, intermediate_wt->min_hz);

        DEBUG_CWT
        {
            TaskTimer("ft->sample_rate = %g", ft->sample_rate).suppressTiming();
            TaskTimer("ft->original_sample_rate = %g", ft->original_sample_rate).suppressTiming();
            TaskTimer("ft->halfs = %u", half_sizes).suppressTiming();
            TaskTimer("intermediate_wt->sample_rate = %g", intermediate_wt->sample_rate).suppressTiming();
            TaskTimer("intermediate_wt->min_hz = %g", intermediate_wt->min_hz).suppressTiming();
            TaskTimer("intermediate_wt->max_hz = %g", intermediate_wt->max_hz).suppressTiming();
        }

        BOOST_ASSERT( intermediate_wt->max_hz <= intermediate_wt->sample_rate/2 );

        ::wtCompute( ft->transform_data->getCudaGlobal().ptr(),
                     intermediate_wt->transform_data->getCudaGlobal().ptr(),
                     intermediate_wt->sample_rate,
                     intermediate_wt->min_hz,
                     intermediate_wt->max_hz,
                     intermediate_wt->transform_data->getNumberOfElements(),
                     1<<half_sizes,
                     _scales_per_octave, sigma() );

        TIME_CWTPART CudaException_ThreadSynchronize();
    }

    {
        // Compute the inverse fourier transform to get the filter banks back
        // in time space
        GpuCpuData<float2>* g = intermediate_wt->transform_data.get();
        cudaExtent n = g->getNumberOfElements();

        intermediate_wt->axis_scale = AxisScale_Logarithmic;

        intermediate_wt->chunk_offset = ft->chunk_offset;

        unsigned time_support = wavelet_time_support_samples( ft->original_sample_rate, intermediate_wt->min_hz );
        time_support >>= half_sizes;
        intermediate_wt->first_valid_sample = time_support;

        if (0==ft->chunk_offset)
            intermediate_wt->first_valid_sample=0;

        DEBUG_CWT {
            TaskTimer("time_support = %u", time_support).suppressTiming();
            TaskTimer("intermediate_wt->first_valid_sample=%u", intermediate_wt->first_valid_sample).suppressTiming();
            TaskTimer("ft->n_valid_samples=%u", ft->n_valid_samples).suppressTiming();
        }

        BOOST_ASSERT( time_support + intermediate_wt->first_valid_sample < ft->n_valid_samples);

        intermediate_wt->n_valid_samples = ft->n_valid_samples - time_support - intermediate_wt->first_valid_sample;

        intermediate_wt->order = Chunk::Order_row_major;

        if (0 /* cpu version */ ) {
            TIME_CWTPART TaskTimer tt("inverse ooura, redundant=%u+%u valid=%u",
                                  intermediate_wt->first_valid_sample,
                                  intermediate_wt->nSamples() - intermediate_wt->n_valid_samples - intermediate_wt->first_valid_sample,
                                  intermediate_wt->n_valid_samples);

            // Move to CPU
            float2* p = g->getCpuMemory();

            pChunk c( new CwtChunk );
            for (unsigned h=0; h<n.height; h++) {
                c->transform_data.reset(
                        new GpuCpuData<float2>(p + n.width*h,
                                       make_cudaExtent(n.width,1,1),
                                       GpuCpuVoidData::CpuMemory, true));
                GpuCpuData<float>* fb = _fft.backward( c )->waveform_data();
                memcpy( p + n.width*h, fb->getCpuMemory(), fb->getSizeInBytes1D() );
            }

            // Move back to GPU
            g->getCudaGlobal( false );
        }
        if (1 /* gpu version */ ) {
            TIME_CWTPART TaskTimer tt("inverse cufft, redundant=%u+%u valid=%u, size(%u, %u)",
                                  intermediate_wt->first_valid_sample,
                                  intermediate_wt->nSamples() - intermediate_wt->n_valid_samples - intermediate_wt->first_valid_sample,
                                  intermediate_wt->n_valid_samples,
                                  n.width, n.height);

            cufftComplex *d = g->getCudaGlobal().ptr();

            //Stft stft;
            //stft.set_exact_chunk_size(n.width);

            {
                CufftHandleContext& fftctx = _fft_many[ n.width*n.height ];
                {
                    //TIME_CWTPART TaskTimer tt("Allocating inverse fft");
                    fftctx(n.width, n.height);
                }

                CufftException_SAFE_CALL(cufftExecC2C(fftctx(n.width, n.height), d, d, CUFFT_INVERSE));
            }

            TIME_CWTPART CudaException_ThreadSynchronize();
        }
    }

    return intermediate_wt;
}


Signal::pBuffer Cwt::
        inverse( pChunk pchunk )
{
    CudaException_CHECK_ERROR();

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
            part->min_hz, part->max_hz) );

        Signal::pBuffer inv = inverse(part);
        Signal::pBuffer super = SuperSample::supersample(inv, pchunk->sample_rate);

        DEBUG_CWT {
            tt->getStream()
                    << "Upsampled inv " << inv->getInterval().toString()
                    << " by factor " << pchunk->sample_rate/inv->sample_rate
                    << " to " << super->getInterval().toString(); tt->flushStream();

            GpuCpuData<float> mdata( part->transform_data->getCpuMemory(),
                                 make_cudaExtent( part->transform_data->getNumberOfElements1D(), 1, 1),
                                 GpuCpuVoidData::CpuMemory, true );
        }

        //TaskInfo("super->getInterval() = %s, first_valid_sample = %u",
        //         super->getInterval().toString().c_str(), part->first_valid_sample);

        *r += *super;
    }

    BOOST_ASSERT( pchunk->getInterval() == r->getInterval() );
    TIME_ICWT {
        STAT_CWT Statistics<float>( r->waveform_data() );
    }

    TIME_ICWT CudaException_ThreadSynchronize();

    return r;
}


Signal::pBuffer Cwt::
        inverse( Tfr::CwtChunkPart* pchunk )
{
    Chunk &chunk = *pchunk;

    cudaExtent x = chunk.transform_data->getNumberOfElements();

    Signal::pBuffer r( new Signal::Buffer(
            chunk.chunk_offset,
            x.width,
            chunk.sample_rate
            ));

    float2* p = chunk.transform_data->getCudaGlobal().ptr();

    if (pchunk->original_sample_rate != pchunk->sample_rate)
    {
        // Skip first row
        p += x.width;
        x.height--;
    }

    ::wtInverse( p,
                 r->waveform_data()->getCudaGlobal().ptr(),
                 x,
                 _stream );

    TIME_ICWT CudaException_ThreadSynchronize();

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
        set_min_hz(float value)
{
    if (value == _min_hz) return;

    _fft_many.clear();

    _min_hz = value;
}


unsigned Cwt::
        nScales(float fs) const
{
    float number_of_octaves = log2(get_max_hz(fs)) - log2(_min_hz);
    return 1 + (unsigned)(number_of_octaves * scales_per_octave());
}


void Cwt::
        scales_per_octave( float value )
{
    if (value==_scales_per_octave) return;

    _fft_many.clear();

    _scales_per_octave=value;
}


void Cwt::
        tf_resolution( float value )
{
    if (value == _tf_resolution) return;

    _fft_many.clear();

    _tf_resolution = value;
}


float Cwt::
        sigma() const
{
    return _scales_per_octave/_tf_resolution;
}


float Cwt::
        compute_frequency2( float fs, float normalized_scale ) const
{
    float start = get_max_hz(fs);
    float steplogsize = log2(get_min_hz(fs)) - log2(get_max_hz(fs));

    float hz = start * exp2((1-normalized_scale) * steplogsize);
    return hz;
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
    // Align to 1<<c upwards
    support_samples = (support_samples + (1<<c) - 1) >> c << c;
    return support_samples;
}


unsigned Cwt::
        next_good_size( unsigned current_valid_samples_per_chunk, float fs )
{
    unsigned r = wavelet_time_support_samples( fs );
    unsigned max_bin = find_bin( nScales( fs ) - 1 );
    if ( 0 == r>>max_bin )
        r = 1 << (max_bin-1);
    unsigned T = r + current_valid_samples_per_chunk + r;
    unsigned nT = spo2g(T);

    if (nT > 1<<19) // For some reason cufft sais CUFFT_ALLOC_FAILED for fft size (1<<20)
        nT = 1<<19;

    if(nT <= 2*r)
        nT = spo2g(2*r);
    unsigned L = nT - 2*r;

    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);

    if (free < required_gpu_bytes(L, fs ))
        return prev_good_size( L, fs );

    return L;
}


unsigned Cwt::
        prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate )
{
    unsigned r = wavelet_time_support_samples( sample_rate );
    unsigned max_bin = find_bin( nScales( sample_rate ) - 1 );
    if ( 0 == r>>max_bin )
        r = 1 << (max_bin-1);
    unsigned T = r + current_valid_samples_per_chunk + r;
    unsigned nT = lpo2s(T);
    if (nT <= 2*r)
    {
        nT = spo2g(2*r);
        return nT - 2*r;
    }
    unsigned L = nT - 2*r;

    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);

    if (free < required_gpu_bytes(L, sample_rate ))
        return prev_good_size( L, sample_rate );

    return L;
}


size_t Cwt::
        required_gpu_bytes(unsigned L, float sample_rate) const
{
    unsigned r = wavelet_time_support_samples( sample_rate );
    unsigned max_bin = find_bin( nScales( sample_rate ) - 1 );
    size_t sum = sizeof(float2)*5*(L+2*r)*nScales( sample_rate )/(1+max_bin)*1.15;
    return sum;
}


unsigned Cwt::
        find_bin( unsigned j ) const
{
#ifdef CWT_NOBINS
    return 0;
#endif

    float v = _scales_per_octave;
    float log2_a = 1.f/v;
    float bin = log2_a * j - log2( 1.f + _wavelet_scale_suppport/(2*M_PI*sigma()) );

    if (bin < 0)
        bin = 0;

    // could take maximum number of bins into account and meld all the last
    // ones into the same bin, effectively making the second last bin all empty
    // unsigned n_j = nScales( fs );

    return floor(bin);
}


void Cwt::
        resetSingleton()
{
    static_singleton.reset();
    gc();
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
