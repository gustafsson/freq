#include "tfr-cwt.h"

#define TIME_CWT

namespace Tfr {

Cwt::Cwt( cudaStream_t stream=0 )
:   _stft( stream ),
    _stream( stream )
{
}

pChunk operator()( Signal::pBuffer buffer )
{
    pStftData ft = _stft( buffer );

    {
        TaskTimer tt(TaskTimer::LogVerbose, "prerequisites");

        cudaExtent requiredWtSz = make_cudaExtent( ft->getNumberOfElements().width, nScales(), 1 );

        if (_intermediate_wt && _intermediate_wt->transform_data->getNumberOfElements() != requiredWtSz)
            gc();

        if (!_intermediate_wt) {
            // allocate a new chunk
            pTransform_chunk chunk = pTransform_chunk ( new Transform_chunk());

            chunk->transform_data.reset(new GpuCpuData<float2>( 0, requiredWtSz, GpuCpuVoidData::CudaGlobal ));
            _intermediate_wt = chunk;
        }


        #ifdef TIME_CWT
            CudaException_ThreadSynchronize();
        #endif
    }

    {
        TaskTimer tt(TaskTimer::LogVerbose, "inflating");
        _intermediate_wt->sample_rate =  buffer->sample_rate;
        _intermediate_wt->min_hz = _min_hz;
        _intermediate_wt->max_hz = max_hz(buffer->sample_rate);

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
        _intermediate_wt->first_valid_sample = 0;
        _intermediate_wt->chunk_offset = 0;
        _intermediate_wt->n_valid_samples = buffer->number_of_samples();

        ::wtCompute( ft->getCudaGlobal().ptr(),
                     _intermediate_wt->transform_data->getCudaGlobal().ptr(),
                     _intermediate_wt->sample_rate,
                     _intermediate_wt->min_hz,
                     _intermediate_wt->max_hz,
                     _intermediate_wt->transform_data->getNumberOfElements(),
                     _scales_per_octave );

        #ifdef TIME_CWT
            CudaException_ThreadSynchronize();
        #endif
    }

    {
        TaskTimer tt(TaskTimer::LogVerbose, "inverse fft");

        // Transform signal back
        GpuCpuData<float2>* g = _intermediate_wt->transform_data.get();
        cudaExtent n = g->getNumberOfElements();
        cufftComplex *d = g->getCudaGlobal().ptr();

        if (_fft_many == (cufftHandle)-1)
            cufftSafeCall(cufftPlan1d(&_fft_many, n.width, CUFFT_C2C, n.height));

        cufftSafeCall(cufftSetStream(_fft_many, stream));
        cufftSafeCall(cufftExecC2C(_fft_many, d, d, CUFFT_INVERSE));

        #ifdef TIME_CWT
            CudaException_ThreadSynchronize();
        #endif
    }

    return _intermediate_wt;
}

void Cwt::min_hz(float value) {
    if (value == _min_hz) return;
    gc();
    _min_hz = value;
}

float Cwt::number_of_octaves( unsigned sample_rate ) const {
    return log2(max_hz(sample_rate)) - log2(_min_hz);
}

void Cwt::scales_per_octave( unsigned ) {
    if (value==_scales_per_octave) return;
    gc();
    _scales_per_octave=value;
}

void Cwt::gc() {
    _intermediate_wt.reset();

    // Destroy CUFFT context
    if (_fft_many == (cufftHandle)-1)
        cufftDestroy(_fft_many);

    _fft_many = -1;
}

} // namespace Tfr
