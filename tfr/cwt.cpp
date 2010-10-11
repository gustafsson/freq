#include "cwt.h"
#include "cwtchunk.h"
#include "stft.h"
#include "wavelet.cu.h"

#include "signal/buffersource.h"
#include "filters/supersample.h"

#include <cufft.h>
#include <throwInvalidArgument.h>
#include <CudaException.h>
#include <neat_math.h>

#include <boost/foreach.hpp>

#ifdef _MSC_VER
#include <msc_stdc.h>
#endif

// #define TIME_CWT if(0)
#define TIME_CWT

// #define TIME_ICWT if(0)
#define TIME_ICWT

namespace Tfr {

Cwt::
        Cwt( float scales_per_octave, float wavelet_time_suppport, cudaStream_t stream )
:   _fft( /*stream*/ ),
    _stream( stream ),
    _min_hz( 20 ),
    _scales_per_octave( scales_per_octave ),
    _tf_resolution( 2.5 ),
//    _fft_many(stream),
    _wavelet_time_suppport( wavelet_time_suppport )
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
    static pTransform P(new Cwt());
    return P;
}


pChunk Cwt::
        operator()( Signal::pBuffer buffer )
{
    std::stringstream ss;
    TIME_CWT TaskTimer tt("Cwt buffer %s, %u samples. [%g, %g] s", ((std::stringstream&)(ss<<buffer->getInterval())).str().c_str(), buffer->number_of_samples(), buffer->start(), buffer->length()+buffer->start() );

    Signal::BufferSource bs( buffer );

    float v = _scales_per_octave;
    unsigned long offset = buffer->sample_offset;
    unsigned std_samples = wavelet_time_support_samples( buffer->sample_rate );
    unsigned long first_valid_sample = std_samples;
    if (0==offset)
        first_valid_sample = 0;

    BOOST_ASSERT( std_samples + first_valid_sample < buffer->number_of_samples());

    unsigned valid_samples = buffer->number_of_samples() - std_samples - first_valid_sample;


    // find all sub chunks
    unsigned prev_j = 0;
    unsigned n_j = nScales( buffer->sample_rate );
    const float log2_a = 1.f / v; // a = 2^(1/v)

    pChunk wt( new CwtChunk() );

    for( unsigned c=0; prev_j<n_j; ++c )
    {
        // find the biggest j that is required to be in this chunk
        unsigned next_j;
        if ((valid_samples>>(c+2)) < 1)
            next_j = n_j;
        else
        {
            next_j = prev_j;

            for (unsigned j = prev_j; j<n_j; ++j)
            {
                float aj = exp2f( log2_a * j );
                float sigmaW0 = valid_samples / (2*aj*M_PI*sigma());
                float center_index = valid_samples / aj;
                float edge = center_index + 4*sigmaW0;

                if ( edge >= (valid_samples >> (c+1) ))
                    next_j = j;
            }
        }

        if (2*(n_j - next_j) < n_j - prev_j)
            next_j = n_j;

        // Move next_j forward one step so that it points to the first 'j'
        // that is not needed in this chunk part
        next_j = std::min(n_j, next_j+1);

        // Include one more 'j' for interpolation between parts
        unsigned stop_j = std::min(n_j, next_j+1);
        float hz = get_max_hz( buffer->sample_rate );
        hz *= exp2f(stop_j/-v);
        unsigned sub_std_samples = wavelet_time_support_samples( buffer->sample_rate, hz );
        sub_std_samples = (sub_std_samples + (1<<c)-1)>>c;
        sub_std_samples = (sub_std_samples + 15)/16*16;
        sub_std_samples<<=c;
        unsigned sub_first_valid = sub_std_samples;
        if (0==offset)
            sub_first_valid = 0;

        BOOST_ASSERT( sub_first_valid <= offset + first_valid_sample );
        unsigned sub_start = offset + first_valid_sample - sub_first_valid;
        unsigned sub_length = sub_first_valid + valid_samples + sub_std_samples;

        Signal::Interval subinterval(sub_start, sub_start + sub_length );
        std::cout << "c=" << c << ", hz=" << hz << ", subinterval=" << subinterval << std::endl;
        pChunk ft = _fft( bs.readFixedLength( subinterval ));
        std::cout << " ft=" << ft->getInterval() << std::endl;

        float bfs = bs.sample_rate();
        float fs = ft->sample_rate;
        unsigned n_scales = stop_j - prev_j;
        ((StftChunk*)ft.get())->setHalfs( c ); // discard most of it if c>0
        ft->chunk_offset >>= c;
        // ft->first_valid_sample = 0 for fft
        ft->n_valid_samples >>= c;
        std::cout << " ft(c)=" << ft->getInterval() << std::endl;

        pChunk chunkpart = computeChunkPart( ft, prev_j, n_scales );
        std::cout << " chunkpart=" << chunkpart->getInterval() << std::endl;
        // TODO verify output
        ((CwtChunk*)wt.get())->chunks.push_back( chunkpart );

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
    std::cout << "wt=" << wt->getInterval() << std::endl;
    return wt;
}


pChunk Cwt::
        computeChunkPart( pChunk ft, unsigned first_scale, unsigned n_scales )
{
    TaskTimer tt("computeChunkPart first_scale=%u, n_scales=%u", first_scale, n_scales);

    pChunk intermediate_wt( new CwtChunkPart() );

    {
        cudaExtent requiredWtSz = make_cudaExtent( ft->nScales(), n_scales, 1 );
        TIME_CWT TaskTimer tt("prerequisites (%u, %u, %u)", requiredWtSz.width, requiredWtSz.height, requiredWtSz.depth);

        // allocate a new chunk
        intermediate_wt->transform_data.reset(new GpuCpuData<float2>( 0, requiredWtSz, GpuCpuVoidData::CudaGlobal ));

        TIME_CWT CudaException_ThreadSynchronize();
    }

    unsigned half_sizes;
    {
        Tfr::StftChunk* stchunk = dynamic_cast<Tfr::StftChunk*>(ft.get());
        half_sizes = stchunk->halfs();
    }

    {        
        TIME_CWT TaskTimer tt("inflating");

        // ft->sample_rate is related to intermediate_wt->sample_rate by
        // intermediate_wt->sample_rate == ft->n_valid_samples * ft->sample_rate
        // (except for numerical errors)
        intermediate_wt->sample_rate = ldexp(ft->original_sample_rate, -half_sizes);
        intermediate_wt->original_sample_rate = ft->original_sample_rate;
        TaskTimer("ft->sample_rate = %g", ft->sample_rate);
        TaskTimer("ft->original_sample_rate = %g", ft->original_sample_rate).suppressTiming();
        TaskTimer("ft->n_valid_samples = %u", ft->n_valid_samples);
        TaskTimer("intermediate_wt->sample_rate = %g", intermediate_wt->sample_rate).suppressTiming();
        TaskTimer("intermediate_wt->original_sample_rate = %g", intermediate_wt->original_sample_rate).suppressTiming();
        TaskTimer("log2(intermediate_wt->sample_rate) = %g", log2(intermediate_wt->sample_rate)).suppressTiming();
        TaskTimer("log2(intermediate_wt->sample_rate) = %g", log2(intermediate_wt->sample_rate)).suppressTiming();

        intermediate_wt->min_hz = get_max_hz(ft->original_sample_rate)*exp2f( (first_scale + n_scales)/-_scales_per_octave );
        intermediate_wt->max_hz = get_max_hz(ft->original_sample_rate)*exp2f( first_scale/-_scales_per_octave );

        TaskTimer("intermediate_wt->sample_rate = %g", intermediate_wt->sample_rate).suppressTiming();
        TaskTimer("intermediate_wt->min_hz = %g", intermediate_wt->min_hz).suppressTiming();
        TaskTimer("intermediate_wt->max_hz = %g", intermediate_wt->max_hz).suppressTiming();

        BOOST_ASSERT( intermediate_wt->max_hz <= intermediate_wt->sample_rate/2 );

        ::wtCompute( ft->transform_data->getCudaGlobal().ptr(),
                     intermediate_wt->transform_data->getCudaGlobal().ptr(),
                     intermediate_wt->sample_rate,
                     intermediate_wt->min_hz,
                     intermediate_wt->max_hz,
                     intermediate_wt->transform_data->getNumberOfElements(),
                     _scales_per_octave, sigma() );

        TIME_CWT CudaException_ThreadSynchronize();
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
        TaskTimer("time_support = %u", time_support).suppressTiming();
        intermediate_wt->first_valid_sample = time_support;

        if (0==ft->chunk_offset)
            intermediate_wt->first_valid_sample=0;

        TaskTimer("ft->n_valid_samples=%u", ft->n_valid_samples).suppressTiming();
        BOOST_ASSERT( time_support + intermediate_wt->first_valid_sample < ft->n_valid_samples);

        intermediate_wt->n_valid_samples = ft->n_valid_samples - time_support - intermediate_wt->first_valid_sample;

        intermediate_wt->order = Chunk::Order_row_major;

        if (0 /* cpu version */ ) {
            TIME_CWT TaskTimer tt("inverse ooura, redundant=%u+%u valid=%u",
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
            TIME_CWT TaskTimer tt("inverse cufft, redundant=%u+%u valid=%u",
                                  intermediate_wt->first_valid_sample,
                                  intermediate_wt->nSamples() - intermediate_wt->n_valid_samples - intermediate_wt->first_valid_sample,
                                  intermediate_wt->n_valid_samples);

            cufftComplex *d = g->getCudaGlobal().ptr();

            //Stft stft;
            //stft.set_exact_chunk_size(n.width);

            {
                CufftHandleContext _fft_many;
                CufftException_SAFE_CALL(cufftExecC2C(_fft_many(n.width, n.height), d, d, CUFFT_INVERSE));
            }

            TIME_CWT CudaException_ThreadSynchronize();
        }
    }

    return intermediate_wt;
}


Signal::pBuffer Cwt::
        inverse( pChunk pchunk )
{
    CudaException_CHECK_ERROR();

    TIME_ICWT TaskTimer tt("InverseCwt(pChunk)");
    Tfr::CwtChunk* cwtchunk = dynamic_cast<Tfr::CwtChunk*>(pchunk.get());
    if (cwtchunk)
        return inverse(cwtchunk);

    Tfr::CwtChunkPart* cwtchunkpart = dynamic_cast<Tfr::CwtChunkPart*>(pchunk.get());
    if (cwtchunkpart)
        return inverse(cwtchunkpart);

    throw std::invalid_argument("Doesn't recognize chunk of type " + demangle( typeid(*pchunk.get()).name()));
}


Signal::pBuffer Cwt::
        inverse( Tfr::CwtChunk* pchunk )
{
    Signal::pBuffer r( new Signal::Buffer(
            pchunk->chunk_offset + pchunk->first_valid_sample,
            pchunk->n_valid_samples,
            pchunk->sample_rate
            ));

    std::cout << "r->getInterval(): " << r->getInterval() << std::endl;
    BOOST_FOREACH( pChunk& part, pchunk->chunks )
    {
        CudaException_CHECK_ERROR();
        Signal::pBuffer inv = inverse(part);
        CudaException_CHECK_ERROR();
        std::cout << "  inv->getInterval(): " << inv->getInterval() << std::endl;
        std::cout << "  inv start = " << inv->sample_offset.asFloat() << std::endl;
        Signal::pBuffer super = Filters::SuperSample::supersample(inv, pchunk->sample_rate);
        std::cout << "  super start = " << super->sample_offset.asFloat() << std::endl;
        std::cout << "  super->getInterval(): " << super->getInterval() << std::endl;

        // ft->chunk_offset = ldexpf(ft->chunk_offset, -c);
        // ft->first_valid_sample = ldexpf(ft->chunk_offset, -c);
        // ft->n_valid_samples = ldexpf(ft->n_valid_samples, -c);
        CudaException_CHECK_ERROR();

        *r += *super;

        CudaException_CHECK_ERROR();
    }
    std::cout << "inverse end" << std::endl;

    return r;
}


Signal::pBuffer Cwt::
        inverse( Tfr::CwtChunkPart* pchunk )
{
    TIME_ICWT TaskTimer tt("InverseCwt(Tfr::CwtChunkPart*)");

    Chunk &chunk = *pchunk;

    Signal::pBuffer r( new Signal::Buffer(
            chunk.chunk_offset + chunk.first_valid_sample,
            chunk.n_valid_samples,
            chunk.sample_rate
            ));

    {
        TIME_ICWT TaskTimer tt(TaskTimer::LogVerbose, __FUNCTION__);
        {
            ::wtInverse( chunk.transform_data->getCudaGlobal().ptr() + chunk.first_valid_sample,
                         r->waveform_data()->getCudaGlobal().ptr(),
                         chunk.transform_data->getNumberOfElements(),
                         chunk.n_valid_samples,
                         _stream );
        }

        TIME_ICWT CudaException_ThreadSynchronize();
    }

    return r;
}

float Cwt::
        get_min_hz( float fs ) const
{
    unsigned n_j = nScales( fs );
    float hz = get_max_hz( fs ) * exp2f(n_j/-_scales_per_octave);
    return hz;
}


void Cwt::
        set_min_hz(float value)
{
    if (value == _min_hz) return;

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

    _scales_per_octave=value;
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
    // 2.5 is Ulfs magic constant
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
        morlet_sigma_t( float fs, float hz ) const
{
    float scale = hz/get_max_hz( fs );
    float j = -_scales_per_octave*log2f(scale);

    const float log2_a = 1.f / _scales_per_octave; // a = 2^(1/v), v = _scales_per_octave
    float aj = exp2f( log2_a * j );

    return aj*sigma();
}


float Cwt::
        morlet_sigma_f( float fs, float hz ) const
{
    float scale = hz/get_max_hz( fs );
    float j = -_scales_per_octave*log2f(scale);

    const float log2_a = 1.f / _scales_per_octave; // a = 2^(1/v), v = _scales_per_octave
    float aj = exp2f( log2_a * j );

    float sigmaW0 = get_max_hz( fs ) / (2*M_PI*aj* sigma() );
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
    unsigned support_samples = morlet_sigma_t( fs, hz ) * _wavelet_time_suppport;
    unsigned c = 1 + log2(get_max_hz(fs)) - log2(hz);
    support_samples = (support_samples + (1<<c) - 1) >> c;
    support_samples = (support_samples+15)/16*16;
    support_samples <<= c;
    return support_samples;
}


unsigned Cwt::
        next_good_size( unsigned current_valid_samples_per_chunk, float fs )
{
    unsigned r = wavelet_time_support_samples( fs );
    unsigned T = r + current_valid_samples_per_chunk + r;
    unsigned nT = spo2g(T);
    if(nT <= 2*r)
        nT = spo2g(2*r);
    return nT - 2*r;
}


unsigned Cwt::
        prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate )
{
    unsigned r = wavelet_time_support_samples( sample_rate );
    unsigned T = r + current_valid_samples_per_chunk + r;
    unsigned nT = lpo2s(T);
    if (nT <= 2*r)
        nT = spo2g(2*r);
    return nT - 2*r;
}

} // namespace Tfr
