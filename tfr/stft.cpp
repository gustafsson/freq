#include "stft.h"

#include <cufft.h>
#include <throwInvalidArgument.h>
#include <CudaException.h>
#include <neat_math.h>

#ifdef _MSC_VER
#include <msc_stdc.h>
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
	} else {
		_creator_thread.throwIfNotSame(__FUNCTION__);
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
		_creator_thread.throwIfNotSame(__FUNCTION__);

		CufftException_SAFE_CALL(cufftDestroy(_handle));

        _handle = 0;
    }
}

Fft::
        Fft(/*cudaStream_t stream*/)
//:   _fft_single( stream )
{
}

Fft::
        ~Fft()
{
}

pFftChunk Fft::
        forward( Signal::pBuffer b)
{
    // cufft is faster for larger ffts, but as the GPU is the bottleneck we can
    // just as well do it on the CPU instead

    // return computeWithCufft(b, -1);
    return computeWithOoura(b,-1);
}

pFftChunk Fft::
        backward( Signal::pBuffer b)
{
    // return computeWithCufft(b,1);
    return computeWithOoura(b,1);
}

//extern "C" { void cdft(int, int, double *); }
extern "C" { void cdft(int, int, double *, int *, double *); }

pFftChunk Fft::
        computeWithOoura( Signal::pBuffer buffer, int direction )
{
    TIME_STFT TaskTimer tt("Fft Ooura");

    unsigned n = buffer->number_of_samples();
    BOOST_ASSERT( 0 < n );
    unsigned N = spo2g(n-1);

    if (q.size() != 2*N) {
        TIME_STFT TaskTimer tt("Resizing buffers");
        q.resize(2*N);
        w.resize(N/2);
        ip.resize(N/2);
        ip[0] = 0;
    } else {
        TIME_STFT TaskTimer("Reusing data").suppressTiming();
    }

    float* p=buffer->waveform_data->getCpuMemory();

    {
        TIME_STFT TaskTimer tt("Converting from float%c to double2", buffer->interleaved() == Signal::Buffer::Interleaved_Complex?'2':'1');

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
    }


    {
        TIME_STFT TaskTimer tt("Computing fft");
        cdft(2*N, direction, &q[0], &ip[0], &w[0]);
    }

    {
        TIME_STFT TaskTimer tt("Converting from double2 to float2");

        pFftChunk intermediate_fft;
        intermediate_fft.reset(new GpuCpuData<float2>( 0, make_cudaExtent( N, 1, 1 ) ));
        p = (float*)intermediate_fft->getCpuMemory();
        for (unsigned i=0; i<2*N; i++)
            p[i] = (float)q[i];

        return intermediate_fft;
    }
}

pFftChunk Fft::
        computeWithCufft( Signal::pBuffer buffer, int direction )
{
    TIME_STFT TaskTimer tt("FFt cufft");
    if (buffer->interleaved() != Signal::Buffer::Interleaved_Complex) {
        TIME_STFT TaskTimer tt("getInterleaved(Complex)");
        buffer = buffer->getInterleaved( Signal::Buffer::Interleaved_Complex );
    }

    unsigned n = buffer->number_of_samples();
    BOOST_ASSERT( 0 < n );
    unsigned N = spo2g(n-1);

    // The in-signal is padded to a power of 2 (cufft manual suggest a "power of a small prime") for faster fft calculations
    cudaExtent required_stft_sz = make_cudaExtent( N, 1, 1 );

    pFftChunk intermediate_fft;
    {
    TIME_STFT TaskTimer ttf("Allocating %u float2", N);
    intermediate_fft.reset(new GpuCpuData<float2>( 0, required_stft_sz, GpuCpuVoidData::CudaGlobal ));
    }

    TIME_STFT TaskTimer ttf("forward fft");
    cufftComplex* d = intermediate_fft->getCudaGlobal().ptr();
    cudaMemset( d, 0, intermediate_fft->getSizeInBytes1D() );
    cudaMemcpy( d,
                buffer->waveform_data->getCudaGlobal().ptr(),
                buffer->waveform_data->getSizeInBytes().width,
                cudaMemcpyDeviceToDevice );

    // Transform signal
	CufftHandleContext _fft_single;
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

    if (0<count)
    {
        unsigned
                slice = count,
                n = 0;

        while(n < count)
        {
            try
            {
                CufftException_SAFE_CALL(cufftPlan1d(&fft_many, chunk_size, CUFFT_C2C, slice));

                CufftException_SAFE_CALL(cufftSetStream(fft_many, stream));
                CufftException_SAFE_CALL(cufftExecC2C(fft_many, &d[n], &d[n], CUFFT_FORWARD));
                cufftDestroy(fft_many);

                n += slice;
            } catch (const CufftException&) {
                if (slice>0)
                    slice/=2;
                else
                    throw;
            }
        }
    }

    // Clean leftovers with 0
    if (b->number_of_samples() % chunk_size != 0) {
        cudaMemset( d + ((b->number_of_samples() / chunk_size)*chunk_size), 0, (b->number_of_samples() % chunk_size)*sizeof(cufftComplex) );
    }

    return b;
}

pStft StftSingleton::
        instance()
{
    static pStft stft( new Stft ());
    return stft;
}

} // namespace Tfr
