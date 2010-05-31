#include "tfr-cwt.h"
#include <cufft.h>
#include "tfr-stft.h"
#include <throwInvalidArgument.h>
#include <CudaException.h>
#include "tfr-wavelet.cu.h"

#ifdef _MSC_VER
#include "msc_stdc.h"
#endif

#define TIME_CWT if(0)
//#define TIME_CWT

namespace Tfr {

Cwt::
        Cwt( float scales_per_octave, float wavelet_std_t, cudaStream_t stream )
:   _fft( stream ),
    _stream( stream ),
    _min_hz( 20 ),
    _scales_per_octave( scales_per_octave ),
    _fft_many(stream),
    _wavelet_std_t( wavelet_std_t )
{
}

pChunk Cwt::
        operator()( Signal::pBuffer buffer )
{
    std::stringstream ss;
    TIME_CWT TaskTimer tt("Cwt buffer %s", ((std::stringstream&)(ss<<buffer->getInterval())).str().c_str() );

    pFftChunk ft ( _fft( buffer ) );

    pChunk intermediate_wt( new Chunk() );

    {
        cudaExtent requiredWtSz = make_cudaExtent( ft->getNumberOfElements().width, nScales(buffer->sample_rate), 1 );
        TIME_CWT TaskTimer tt("prerequisites (%u, %u, %u)", requiredWtSz.width, requiredWtSz.height, requiredWtSz.depth);

        // allocate a new chunk
        intermediate_wt->transform_data.reset(new GpuCpuData<float2>( 0, requiredWtSz, GpuCpuVoidData::CudaGlobal ));

        TIME_CWT CudaException_ThreadSynchronize();
    }

    {
        TIME_CWT TaskTimer tt("inflating");
        intermediate_wt->sample_rate =  buffer->sample_rate;
        intermediate_wt->min_hz = _min_hz;
        intermediate_wt->max_hz = max_hz(buffer->sample_rate);

        ::wtCompute( ft->getCudaGlobal().ptr(),
                     intermediate_wt->transform_data->getCudaGlobal().ptr(),
                     intermediate_wt->sample_rate,
                     intermediate_wt->min_hz,
                     intermediate_wt->max_hz,
                     intermediate_wt->transform_data->getNumberOfElements(),
                     _scales_per_octave );

        TIME_CWT CudaException_ThreadSynchronize();
    }

    {
        // Transform signal back
        GpuCpuData<float2>* g = intermediate_wt->transform_data.get();
        cudaExtent n = g->getNumberOfElements();

        if (0 /* cpu version */ ) {
            TIME_CWT TaskTimer tt("inverse ooura");

            // Move to CPU
            float2* p = g->getCpuMemory();

            Signal::pBuffer b( new Signal::Buffer(Signal::Buffer::Interleaved_Complex) );
            for (unsigned h=0; h<n.height; h++) {
                b->waveform_data.reset(
                        new GpuCpuData<float>(p + n.width*h,
                                       make_cudaExtent(2*n.width,1,1),
                                       GpuCpuVoidData::CpuMemory, true));
                pFftChunk fc = _fft.backward( b );
                memcpy( p + n.width*h, fc->getCpuMemory(), fc->getSizeInBytes1D() );
            }

            // Move back to GPU
            g->getCudaGlobal( false );
        }
        if (1 /* gpu version */ ) {
            TIME_CWT TaskTimer tt("inverse cufft");

            cufftComplex *d = g->getCudaGlobal().ptr();

            CufftException_SAFE_CALL(cufftExecC2C(_fft_many(n.width, n.height), d, d, CUFFT_INVERSE));

            TIME_CWT CudaException_ThreadSynchronize();
        }
        intermediate_wt->chunk_offset = buffer->sample_offset;
        intermediate_wt->first_valid_sample = wavelet_std_samples( buffer->sample_rate );
        if (0==buffer->sample_offset)
            intermediate_wt->first_valid_sample=0;
        intermediate_wt->max_hz = max_hz( buffer->sample_rate );
        intermediate_wt->min_hz = min_hz();

        BOOST_ASSERT(wavelet_std_samples( buffer->sample_rate ) + intermediate_wt->first_valid_sample < buffer->number_of_samples());

        intermediate_wt->n_valid_samples = buffer->number_of_samples() - wavelet_std_samples( buffer->sample_rate ) - intermediate_wt->first_valid_sample;

        intermediate_wt->sample_rate = buffer->sample_rate;
    }

    return intermediate_wt;
}

void Cwt::
        min_hz(float value)
{
    if (value == _min_hz) return;

    _min_hz = value;
}

float Cwt::
        number_of_octaves( unsigned sample_rate ) const
{
    return log2(max_hz(sample_rate)) - log2(_min_hz);
}

void Cwt::
        scales_per_octave( unsigned value)
{
    if (value==_scales_per_octave) return;

    _scales_per_octave=value;
}

unsigned Cwt::
        wavelet_std_samples( unsigned sample_rate ) const
{
    return (_wavelet_std_t*sample_rate+31)/32*32;
}

pChunk CwtSingleton::
        operate( Signal::pBuffer b )
{
    return (*instance())( b );
}

pCwt CwtSingleton::
        instance()
{
    static pCwt cwt( new Cwt ());
    return cwt;
}

} // namespace Tfr
