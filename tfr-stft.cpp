#include "tfr-stft.h"
#include <cufft.h>
#include <throwInvalidArgument.h>
#include <CudaException.h>

#ifdef _MSC_VER
#include "msc_stdc.h"
#endif

//#define TIME_STFT
#define TIME_STFT if(0)

namespace Tfr {

CufftHandleContext::
        CufftHandleContext( cudaStream_t stream )
:   _handle(0),
    _stream(stream)
{}

CufftHandleContext::
        ~CufftHandleContext()
{
    destroy();
    _creator_thread.reset();
}

cufftHandle CufftHandleContext::
        operator()( unsigned elems, unsigned batch_size )
{
    if (0 == _handle || _elems != elems || _batch_size != batch_size) {
        _elems = elems;
        _batch_size = batch_size;
        create();
    }
    return _handle;
}

void CufftHandleContext::
        create()
{
    destroy();
    CufftException_SAFE_CALL(cufftPlan1d(&_handle, _elems, CUFFT_C2C, _batch_size));
    CufftException_SAFE_CALL(cufftSetStream(_handle, _stream ));
    _creator_thread.reset();
}

void CufftHandleContext::
        destroy()
{
    if (_handle) {
        if (_creator_thread.isSameThread())
            CufftException_SAFE_CALL(cufftDestroy(_handle));
        _handle = 0;
    }
}

Fft::
        Fft(cudaStream_t stream)
:   _fft_single( stream )
{
}

Fft::
        ~Fft()
{
}

pFftChunk Fft::
        forward( Signal::pBuffer b)
{
//    computeWithCufft(b);
    return computeWithOoura(b,-1);
}

pFftChunk Fft::
        backward( Signal::pBuffer b)
{
//    computeWithCufft(b);
    return computeWithOoura(b,1);
}

//extern "C" { void cdft(int, int, double *); }
extern "C" { void cdft(int, int, double *, int *, double *); }

pFftChunk Fft::
        computeWithOoura( Signal::pBuffer buffer, int direction )
{
    TIME_STFT TaskTimer tt("Fft Ooura");

    unsigned n = buffer->number_of_samples();
    unsigned N = (1 << ((unsigned)ceil(log2((float)n))));

    if (q.size() != 2*N) {
        q.resize(2*N);
        w.resize(N/2);
        ip.resize(N/2);
        ip[0] = 0;
    }

    float* p=buffer->waveform_data->getCpuMemory();
    if (buffer->interleaved() == Signal::Buffer::Interleaved_Complex) {
        for (unsigned i=0; i<2*n; i++)
            q[i] = p[i];
    } else {
        for (unsigned i=0; i<n; i++)
        {
            q[2*i + 0] = p[i];
            q[2*i + 1] = 0;
        }
    }

    for (unsigned i=2*n; i<2*N; i++)
        q[i] = 0;



    cdft(2*N, direction, &q[0], &ip[0], &w[0]);



    pFftChunk intermediate_fft;
    intermediate_fft.reset(new GpuCpuData<float2>( 0, make_cudaExtent( N, 1, 1 ) ));
    p = (float*)intermediate_fft->getCpuMemory();
    for (unsigned i=0; i<2*N; i++)
        p[i] = q[i];

    return intermediate_fft;
}

pFftChunk Fft::
        computeWithCufft( Signal::pBuffer buffer, int direction )
{
    TIME_STFT TaskTimer tt("FFt cufft");
    if (buffer->interleaved() != Signal::Buffer::Interleaved_Complex) {
        TIME_STFT TaskTimer tt("getInterleaved(Complex)");
        buffer = buffer->getInterleaved( Signal::Buffer::Interleaved_Complex );
    }

    pFftChunk intermediate_fft;

    cudaExtent required_stft_sz = make_cudaExtent( buffer->waveform_data->getNumberOfElements().width/2, 1, 1 );
    // The in-signal is padded to a power of 2 (or rather, "power of a small prime") for faster fft calculations
    required_stft_sz.width = (1 << ((unsigned)ceil(log2((float)required_stft_sz.width))));

    intermediate_fft.reset(new GpuCpuData<float2>( 0, required_stft_sz, GpuCpuVoidData::CudaGlobal ));


    TIME_STFT TaskTimer ttf("forward fft");
    cufftComplex* d = intermediate_fft->getCudaGlobal().ptr();
    cudaMemset( d, 0, intermediate_fft->getSizeInBytes1D() );
    cudaMemcpy( d,
                buffer->waveform_data->getCudaGlobal().ptr(),
                buffer->waveform_data->getSizeInBytes().width,
                cudaMemcpyDeviceToDevice );

    // Transform signal
    CufftException_SAFE_CALL(cufftExecC2C(
            _fft_single(intermediate_fft->getNumberOfElements().width, 1),
            d, d,
            direction==-1?CUFFT_FORWARD:CUFFT_INVERSE));

    TIME_STFT CudaException_ThreadSynchronize();

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
        cudaMemset( d + ((b->number_of_samples() / chunk_size)*chunk_size), 0, (b->number_of_samples() % chunk_size)*sizeof(cufftComplex) );
    }

    return b;
}

} // namespace Tfr
