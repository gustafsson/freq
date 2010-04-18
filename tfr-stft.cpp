#include "tfr-stft.h"

#define TIME_STFT

namespace Tfr {

Stft::Stft(cudaStream_t stream)
:   _stream(stream),
    _fft_single(-1)
{
}

Stft::~Stft() {
    gc();
}

pStftData stft::operator()( Signal::pBuffer buffer )
{
    if (buffer->interleaved != Signal::Buffer::Interleaved_Complex) {
        buffer = buffer->getInterleaved( Signal::Buffer::Interleaved_Complex );
    }

    cudaExtent required_stft_sz = make_cudaExtent( buffer->waveform_data->getNumberOfElements().width/2, 1, 1 );
    // The in-signal is padded to a power of 2 (or rather, "power of a small prime") for faster fft calculations
    required_stft_sz.width = (1 << ((unsigned)ceil(log2((float)required_stft_sz.width))));

    if (_intermediate_stft && _intermediate_stft->getNumberOfElements()!=required_stft_sz)
        gc();

    if (!_intermediate_stft)
        _intermediate_stft.reset(new GpuCpuData<float2>( 0, required_stft_sz, GpuCpuVoidData::CudaGlobal ));


    TaskTimer tt(TaskTimer::LogVerbose, "forward fft");
    cufftComplex* d = _intermediate_stft->getCudaGlobal().ptr();
    cudaMemset( d, 0, _intermediate_stft->getSizeInBytes1D() );
    cudaMemcpy( d,
                buffer->waveform_data->getCudaGlobal().ptr(),
                buffer->waveform_data->getSizeInBytes().width,
                cudaMemcpyDeviceToDevice );

    // Transform signal
    if (_fft_single == (cufftHandle)-1)
        cufftSafeCall(cufftPlan1d(&_fft_single, _intermediate_ft->getNumberOfElements().width, CUFFT_C2C, 1));

    cufftSafeCall(cufftSetStream(_fft_single, _stream));
    cufftSafeCall(cufftExecC2C(_fft_single, d, d, CUFFT_FORWARD));

    #ifdef TIME_STFT
        CudaException_ThreadSynchronize();
    #endif

    return _intermediate_stft;
}

void gc() {
    _intermediate_stft.reset();

    // Destroy CUFFT context
    if (_fft_single == (cufftHandle)-1)
        cufftDestroy(_fft_single);

    _fft_single = -1;
}
} // namespace Tfr
