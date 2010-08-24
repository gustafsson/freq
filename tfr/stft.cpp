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

pChunk Fft::
        forward( Signal::pBuffer b)
{
    // cufft is faster for larger ffts, but as the GPU is the bottleneck we can
    // just as well do it on the CPU instead

    if (b->interleaved() != Signal::Buffer::Interleaved_Complex)
    {
        // TODO implement void computeWithOoura( GpuCpuData<double2>& input, GpuCpuData<double2>& output, int direction );
        b = b->getInterleaved( Signal::Buffer::Interleaved_Complex );
    }

    cudaExtent input_n = b->waveform_data->getNumberOfElements();
    cudaExtent output_n = input_n;

    // The in-signal is padded to a power of 2 (cufft manual suggest a "power
    // of a small prime") for faster fft calculations
    output_n.width = spo2g( output_n.width / 2 - 1 );

    pChunk chunk( new Chunk );
    chunk->transform_data.reset( new GpuCpuData<float2>(
            0,
            output_n,
            GpuCpuVoidData::CpuMemory ));

    input_n.width /= 2;
    GpuCpuData<float2> input(
            b->waveform_data->getCpuMemory(),
            input_n,
            GpuCpuVoidData::CpuMemory,
            true );

    // computeWithCufft( input, *chunk->transform_data, -1);
    computeWithOoura( input, *chunk->transform_data, -1 );

    chunk->axis_scale = AxisScale_Linear;
    chunk->chunk_offset = b->sample_offset;
    chunk->first_valid_sample = 0;
    chunk->max_hz = b->sample_rate / 2;
    chunk->min_hz = chunk->max_hz / chunk->nSamples();
    chunk->n_valid_samples = chunk->nSamples();
    chunk->order = Chunk::Order_column_major;
    chunk->sample_rate = b->sample_rate / chunk->nSamples();

    return chunk;
}

Signal::pBuffer Fft::
        backward( pChunk chunk)
{
    cudaExtent output_n = chunk->transform_data->getNumberOfElements();

    // The in-signal is padded to a power of 2 (cufft manual suggest a "power
    // of a small prime") for faster fft calculations
    output_n.width = spo2g( output_n.width - 1 );

    Signal::pBuffer ret( new Signal::Buffer( Signal::Buffer::Interleaved_Complex) );
    ret->waveform_data.reset( new GpuCpuData<float>(
            0,
            output_n,
            GpuCpuVoidData::CpuMemory ));

    cudaExtent output_n2 = output_n;
    output_n2.width *= 2;

    GpuCpuData<float2> output(
                ret->waveform_data->getCpuMemory(),
                output_n2,
                GpuCpuVoidData::CpuMemory,
                true );

    // chunk = computeWithCufft(*chunk->transform_data, output, 1);
    computeWithOoura(*chunk->transform_data, output, 1);

    // output shares ptr with ret
    ret->sample_offset = chunk->chunk_offset;
    ret->sample_rate = chunk->sample_rate * chunk->nSamples();

    return ret;
}

// TODO translate cdft to take floats instead of doubles
//extern "C" { void cdft(int, int, double *); }
extern "C" { void cdft(int, int, double *, int *, double *); }

void Fft::
        computeWithOoura( GpuCpuData<float2>& input, GpuCpuData<float2>& output, int direction )
{
    TIME_STFT TaskTimer tt("Fft Ooura");

    unsigned n = input.getNumberOfElements().width;
    unsigned N = output.getNumberOfElements().width;

    if (q.size() != 2*N) {
        TIME_STFT TaskTimer tt("Resizing buffers");
        q.resize(2*N);
        w.resize(N/2);
        ip.resize(N/2);
        ip[0] = 0;
    } else {
        TIME_STFT TaskTimer("Reusing data").suppressTiming();
    }

    float* p = (float*)input.getCpuMemory();

    {
        TIME_STFT TaskTimer tt("Converting from float2 to double2" );

        for (unsigned i=0; i<2*n; i++)
            q[i] = p[i];

        for (unsigned i=2*n; i<2*N; i++)
            q[i] = 0;
    }

    /*TODO remove {
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
    }*/


    {
        TIME_STFT TaskTimer tt("Computing fft");
        cdft(2*N, direction, &q[0], &ip[0], &w[0]);
    }

    {
        TIME_STFT TaskTimer tt("Converting from double2 to float2");

        p = (float*)output.getCpuMemory();
        for (unsigned i=0; i<2*N; i++)
            p[i] = (float)q[i];
    }
}


void Fft::
        computeWithCufft( GpuCpuData<float2>& input, GpuCpuData<float2>& output, int direction )
{
    TIME_STFT TaskTimer tt("FFt cufft");

    cufftComplex* d = output.getCudaGlobal().ptr();
    cudaMemset( d, 0, output.getSizeInBytes1D() );
    cudaMemcpy( d,
                input.getCudaGlobal().ptr(),
                input.getSizeInBytes().width,
                cudaMemcpyDeviceToDevice );

    // Transform signal
    CufftHandleContext _fft_single;
    CufftException_SAFE_CALL(cufftExecC2C(
        _fft_single(output.getNumberOfElements().width, 1),
        d, d,
        direction==-1?CUFFT_FORWARD:CUFFT_INVERSE));

    TIME_STFT CudaException_ThreadSynchronize();
}


/// STFT


Stft::
        Stft( cudaStream_t stream )
:   chunk_size( 1<<11 ),
    _stream( stream )
//    _fft_many( -1 )
{
}


// static
Stft& Stft::
        Singleton()
{
    return *dynamic_cast<Stft*>(SingletonP().get());
}


// static
pTransform Stft::
        SingletonP()
{
    static pTransform P(new Stft());
    return P;
}


Tfr::pChunk Stft::
        operator() (Signal::pBuffer b)
{
    const unsigned stream = 0;

    if (b->interleaved() != Signal::Buffer::Interleaved_Complex)
    {
        b = b->getInterleaved( Signal::Buffer::Interleaved_Complex );
    }

    Tfr::pChunk chunk( new Tfr::Chunk() );
    cudaExtent n = b->waveform_data->getNumberOfElements();
    n.width /= 2; // convert from float to float2

    chunk->transform_data.reset( new GpuCpuData<float2>(
            0,
            n,
            GpuCpuVoidData::CudaGlobal ));


    cufftComplex* input = (cufftComplex*)b->waveform_data->getCudaGlobal().ptr();
    cufftComplex* output = (cufftComplex*)chunk->transform_data->getCudaGlobal().ptr();

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
                CufftException_SAFE_CALL(cufftExecC2C(fft_many, &input[n], &output[n], CUFFT_FORWARD));
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
    if (chunk->nSamples() % chunk_size != 0) {
        cudaMemset( output + ((chunk->nSamples() / chunk_size)*chunk_size), 0, (chunk->nSamples() % chunk_size)*sizeof(cufftComplex) );
    }

    chunk->axis_scale = AxisScale_Linear;
    chunk->chunk_offset = b->sample_offset;
    chunk->first_valid_sample = 0;
    chunk->max_hz = b->sample_rate / 2;
    chunk->min_hz = chunk->max_hz / chunk_size;
    chunk->n_valid_samples = (chunk->nSamples() / chunk_size)*chunk_size;
    chunk->order = Chunk::Order_column_major;
    chunk->sample_rate = b->sample_rate / (float)chunk_size;

    return chunk;
}

} // namespace Tfr
