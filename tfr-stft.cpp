#include "tfr-stft.h"
#include <cufft.h>
#include <throwInvalidArgument.h>
#include <CudaException.h>

#define TIME_STFT

namespace Tfr {

Fft::
        Fft(cudaStream_t stream)
:   _stream(stream)
{
}

Fft::
        ~Fft()
{
}

pFftChunk Fft::
        operator()( Signal::pBuffer buffer )
{
    if (buffer->interleaved() != Signal::Buffer::Interleaved_Complex) {
        buffer = buffer->getInterleaved( Signal::Buffer::Interleaved_Complex );
    }

    pFftChunk intermediate_fft;

    cudaExtent required_stft_sz = make_cudaExtent( buffer->waveform_data->getNumberOfElements().width/2, 1, 1 );
    // The in-signal is padded to a power of 2 (or rather, "power of a small prime") for faster fft calculations
    required_stft_sz.width = (1 << ((unsigned)ceil(log2((float)required_stft_sz.width))));

    intermediate_fft.reset(new GpuCpuData<float2>( 0, required_stft_sz, GpuCpuVoidData::CudaGlobal ));


    TaskTimer tt(TaskTimer::LogVerbose, "forward fft");
    cufftComplex* d = intermediate_fft->getCudaGlobal().ptr();
    cudaMemset( d, 0, intermediate_fft->getSizeInBytes1D() );
    cudaMemcpy( d,
                buffer->waveform_data->getCudaGlobal().ptr(),
                buffer->waveform_data->getSizeInBytes().width,
                cudaMemcpyDeviceToDevice );

    // Transform signal
    cufftHandle fft_single;
    CufftException_SAFE_CALL(cufftPlan1d(&fft_single, intermediate_fft->getNumberOfElements().width, CUFFT_C2C, 1));

    CufftException_SAFE_CALL(cufftSetStream(fft_single, _stream));
    CufftException_SAFE_CALL(cufftExecC2C(fft_single, d, d, CUFFT_FORWARD));
    cufftDestroy(fft_single);

    #ifdef TIME_STFT
        CudaException_ThreadSynchronize();
    #endif

    return intermediate_fft;
}


/// STFT


Stft::
        Stft( cudaStream_t stream )
:   chunk_size( 1<<11 ),
    _stream( stream )
//    _fft_many( -1 )
{
}

Signal::pBuffer Stft::
        operator() (Signal::pBuffer b)
{
    const unsigned stream = 0;

    b = b->getInterleaved( Signal::Buffer::Interleaved_Complex );

    cufftComplex* d = (cufftComplex*)b->waveform_data->getCudaGlobal().ptr();

    // Transform signal
    cufftHandle fft_many;
    unsigned count = b->number_of_samples();
    count/=chunk_size;
    if (0<count) {
        CufftException_SAFE_CALL(cufftPlan1d(&fft_many, chunk_size, CUFFT_C2C, count));

        CufftException_SAFE_CALL(cufftSetStream(fft_many, stream));
        CufftException_SAFE_CALL(cufftExecC2C(fft_many, d, d, CUFFT_FORWARD));
        cufftDestroy(fft_many);
    }

    // Clean leftovers with 0
    if (b->number_of_samples() % chunk_size != 0) {
        cudaMemset( d + ((b->number_of_samples() / chunk_size)*chunk_size), 0, b->number_of_samples() % chunk_size );
    }

    return b;
}

} // namespace Tfr
