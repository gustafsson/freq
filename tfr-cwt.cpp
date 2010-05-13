#include "tfr-cwt.h"
#include <cufft.h>
#include "tfr-stft.h"
#include <throwInvalidArgument.h>
#include <CudaException.h>
#include "tfr-wavelet.cu.h"

#define TIME_CWT

namespace Tfr {

Cwt::
        Cwt( float scales_per_octave, float wavelet_std_t, cudaStream_t stream )
:   _fft( stream ),
    _stream( stream ),
    _min_hz( 20 ),
    _scales_per_octave( scales_per_octave ),
    _wavelet_std_t( wavelet_std_t )
{
}

pChunk Cwt::
        operator()( Signal::pBuffer buffer )
{
    pFftChunk ft ( _fft( buffer ) );

    pChunk intermediate_wt( new Chunk() );

    {
        TaskTimer tt(TaskTimer::LogVerbose, "prerequisites");

        cudaExtent requiredWtSz = make_cudaExtent( ft->getNumberOfElements().width, nScales(buffer->sample_rate), 1 );

        // allocate a new chunk
        intermediate_wt->transform_data.reset(new GpuCpuData<float2>( 0, requiredWtSz, GpuCpuVoidData::CudaGlobal ));

        #ifdef TIME_CWT
            CudaException_ThreadSynchronize();
        #endif
    }

    {
        TaskTimer tt(TaskTimer::LogVerbose, "inflating");
        intermediate_wt->sample_rate =  buffer->sample_rate;
        intermediate_wt->min_hz = _min_hz;
        intermediate_wt->max_hz = max_hz(buffer->sample_rate);

        /*unsigned
            first_valid = _samples_per_chunk*n,
            offs;

        if (first_valid > _wavelet_std_samples)
            offs = first_valid - _wavelet_std_samples;
        else
            offs = 0;

        first_valid-=offs;

        _intermediate_wt->first_valid_sample = first_valid;
        _intermediate_wt->chunk_offset = offs;
        _intermediate_wt->n_valid_samples = _samples_per_chunk;
        */
        intermediate_wt->first_valid_sample = 0;
        intermediate_wt->chunk_offset = 0;
        intermediate_wt->n_valid_samples = buffer->number_of_samples();

        ::wtCompute( ft->getCudaGlobal().ptr(),
                     intermediate_wt->transform_data->getCudaGlobal().ptr(),
                     intermediate_wt->sample_rate,
                     intermediate_wt->min_hz,
                     intermediate_wt->max_hz,
                     intermediate_wt->transform_data->getNumberOfElements(),
                     _scales_per_octave );

        #ifdef TIME_CWT
            CudaException_ThreadSynchronize();
        #endif
    }

    {
        TaskTimer tt(TaskTimer::LogVerbose, "inverse fft");

        // Transform signal back
        GpuCpuData<float2>* g = intermediate_wt->transform_data.get();
        cudaExtent n = g->getNumberOfElements();
        cufftComplex *d = g->getCudaGlobal().ptr();

        cufftHandle     fft_many;
        CufftException_SAFE_CALL(cufftPlan1d(&fft_many, n.width, CUFFT_C2C, n.height));

        CufftException_SAFE_CALL(cufftSetStream(fft_many, _stream));
        CufftException_SAFE_CALL(cufftExecC2C(fft_many, d, d, CUFFT_INVERSE));
        cufftDestroy(fft_many);

        intermediate_wt->chunk_offset = buffer->sample_offset;
        intermediate_wt->first_valid_sample = wavelet_std_samples( buffer->sample_rate );
        intermediate_wt->max_hz = max_hz( buffer->sample_rate );
        intermediate_wt->min_hz = min_hz();

        if (2*intermediate_wt->first_valid_sample >= buffer->number_of_samples())
            ThrowInvalidArgument( _wavelet_std_t );
        else
            intermediate_wt->n_valid_samples = buffer->number_of_samples() - 2*wavelet_std_samples( buffer->sample_rate );

        intermediate_wt->sample_rate = buffer->sample_rate;

        #ifdef TIME_CWT
            CudaException_ThreadSynchronize();
        #endif
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
