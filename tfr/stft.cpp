#include "stft.h"
#include "complexbuffer.h"
#include "signal/buffersource.h"

#include <cufft.h>
#include <throwInvalidArgument.h>
#include <CudaException.h>
#include <neat_math.h>
#include <CudaProperties.h>

#ifdef _MSC_VER
#include <msc_stdc.h>
#endif

//#define TIME_STFT
#define TIME_STFT if(0)

using namespace boost::posix_time;
using namespace boost;

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

    if (_elems == 0 || _batch_size==0)
        return;

    int n = _elems;
    cufftResult r = cufftPlanMany(
            &_handle,
            1,
            &n,
            NULL, 1, 0,
            NULL, 1, 0,
            CUFFT_C2C,
            _batch_size);

    if (CUFFT_SUCCESS != r)
    {
        TaskInfo ti("cufftPlanMany( n = %d, _batch_size = %u ) -> %s",
                    n, _batch_size, CufftException::getErrorString(r));
        size_t free=0, total=0;
        cudaMemGetInfo(&free, &total);
        ti.tt().info("Free mem = %g MB, total = %g MB", free/1024.f/1024.f, total/1024.f/1024.f);
        CufftException_SAFE_CALL( r );
    }

    CufftException_SAFE_CALL(cufftSetStream(_handle, _stream ));
    _creator_thread.reset();
}


void CufftHandleContext::
        destroy()
{
    if (_handle!=0) {
        _creator_thread.throwIfNotSame(__FUNCTION__);

        if (_handle==(cufftHandle)-1)
            TaskInfo("CufftHandleContext::destroy, _handle==(cufftHandle)-1");
        else
        {
            cufftResult errorCode = cufftDestroy(_handle);
            if (errorCode != CUFFT_SUCCESS)
                TaskInfo("CufftHandleContext::destroy, %s", CufftException::getErrorString(errorCode) );
        }

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
        forward( Signal::pBuffer real_buffer)
{
    // cufft is faster for larger ffts, but as the GPU is the bottleneck we can
    // just as well offload it some and do it on the CPU instead

    ComplexBuffer b( *real_buffer );

    cudaExtent input_n = b.complex_waveform_data()->getNumberOfElements();
    cudaExtent output_n = input_n;

    // The in-signal is padded to a power of 2 (cufft manual suggest a "power
    // of a small prime") for faster fft calculations
    output_n.width = spo2g( output_n.width - 1 );

    pChunk chunk( new StftChunk );
    chunk->transform_data.reset( new GpuCpuData<float2>(
            0,
            output_n,
            GpuCpuVoidData::CpuMemory ));

    GpuCpuData<float2>* input = b.complex_waveform_data();

    // TODO choose method based on data size
    computeWithCufft( *input, *chunk->transform_data, -1);
    //computeWithOoura( *input, *chunk->transform_data, -1 );

    chunk->freqAxis.setLinear( b.sample_rate, chunk->nScales() );

    chunk->order = Chunk::Order_column_major;
    chunk->chunk_offset = b.sample_offset;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = input_n.width;
    chunk->sample_rate = b.sample_rate / chunk->n_valid_samples;
    ((StftChunk*)chunk.get())->original_sample_rate = real_buffer->sample_rate;
    return chunk;
}


Signal::pBuffer Fft::
        backward( pChunk chunk)
{
    unsigned scales = chunk->nScales();
    float fs = chunk->sample_rate;
    ComplexBuffer buffer( 0, scales, fs * scales );

    computeWithCufft(*chunk->transform_data, *buffer.complex_waveform_data(), 1);
    // computeWithOoura(*chunk->transform_data, *buffer.complex_waveform_data(), 1);

    Signal::pBuffer r = buffer.get_real();
    if ( r->number_of_samples() != chunk->n_valid_samples )
        r = Signal::BufferSource(r).readFixedLength( Signal::Interval(0, chunk->n_valid_samples ));

    r->sample_offset = chunk->chunk_offset;

    return r;
}


// TODO translate cdft to take floats instead of doubles
//extern "C" { void cdft(int, int, double *); }
extern "C" { void cdft(int, int, double *, int *, double *); }
// extern "C" { void cdft(int, int, float *, int *, float *); }

void Fft::
        computeWithOoura( GpuCpuData<float2>& input, GpuCpuData<float2>& output, int direction )
{
    TIME_STFT TaskTimer tt("Fft Ooura");

    unsigned n = input.getNumberOfElements().width;
    unsigned N = output.getNumberOfElements().width;

    if (-1 != direction)
        BOOST_ASSERT( n == N );

    int magic = 12345678;
    bool vector_length_test = true;

    if (q.size() != 2*N) {
        TIME_STFT TaskTimer tt("Resizing buffers");
        q.resize(2*N + vector_length_test);
        w.resize(N/2 + vector_length_test);
        ip.resize(2+(1<<(int)(log2(N+0.5)-1)) + vector_length_test);

        if (vector_length_test)
        {
            q.back() = magic;
            w.back() = magic;
            ip.back() = magic;
        }
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


    {
        TIME_STFT TaskTimer tt("Computing fft");
        cdft(2*N, direction, &q[0], &ip[0], &w[0]);

        if (vector_length_test)
        {
            BOOST_ASSERT(q.back() == magic);
            BOOST_ASSERT(ip.back() == magic);
            BOOST_ASSERT(w.back() == magic);
        }
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

    // TODO require that input and output is of the exact same size. Do padding
    // before calling computeWithCufft. Then use an out-of-place transform
    // instead of copying the entire input first.

    // Transform signal
    CufftHandleContext _fft_single;
    CufftException_SAFE_CALL(cufftExecC2C(
        _fft_single(output.getNumberOfElements().width, 1),
        d, d,
        direction==-1?CUFFT_FORWARD:CUFFT_INVERSE));

    TIME_STFT CudaException_ThreadSynchronize();
}


/// STFT

std::vector<unsigned> Stft::_ok_chunk_sizes;

Stft::
        Stft( cudaStream_t stream )
:   _stream( stream ),
    _handle_ctx(stream),
    _chunk_size( 1<<11 )
//    _fft_many( -1 )
{
}


Tfr::pChunk Stft::
        operator() (Signal::pBuffer breal)
{
    ComplexBuffer b(*breal);

    BOOST_ASSERT( 0!=_chunk_size );

    cudaExtent n = make_cudaExtent(
            _chunk_size,
            b.number_of_samples()/_chunk_size,
            1 );

    if (0==n.height || 32768<n.height)
        return Tfr::pChunk();


    Tfr::pChunk chunk( new Tfr::StftChunk() );

    chunk->transform_data.reset( new GpuCpuData<float2>(
            0,
            n,
            GpuCpuVoidData::CudaGlobal ));


    cufftComplex* input = (cufftComplex*)b.complex_waveform_data()->getCudaGlobal().ptr();
    cufftComplex* output = (cufftComplex*)chunk->transform_data->getCudaGlobal().ptr();

    // Transform signal
    unsigned count = n.height;

    unsigned
            slices = n.height,
            i = 0;

    // check for available memory
    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);
    free /= 2; // Don't even try to get close to use all memory
    // never use more than 64 MB
    if (free > 64<<20)
        free = 64<<20;
    if (slices * _chunk_size*2*sizeof(cufftComplex) > free)
    {
        slices = free/(_chunk_size*2*sizeof(cufftComplex));
        slices = std::min(512u, std::min((unsigned)n.height, slices));
    }

    while(i < count)
    {
        slices = std::min(slices, count-i);

        try
        {
            CufftException_SAFE_CALL(cufftExecC2C(
                    _handle_ctx(_chunk_size, slices),
                    input + i*n.width,
                    output + i*n.width,
                    CUFFT_FORWARD));

            i += slices;
        } catch (const CufftException& /*x*/) {
            _handle_ctx(0,0);
            if (slices>1)
                slices/=2;
            else
                throw;
        }
    }

    chunk->freqAxis.setLinear( breal->sample_rate, chunk->nScales() );
    chunk->order = Chunk::Order_column_major;
    chunk->chunk_offset = b.sample_offset;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = chunk->nSamples() * chunk->nScales();
    chunk->sample_rate = b.sample_rate / chunk->nScales();
    ((StftChunk*)chunk.get())->original_sample_rate = breal->sample_rate;

    return chunk;
}


//static unsigned absdiff(unsigned a, unsigned b)
//{
//    return a < b ? b - a : a - b;
//}


unsigned Stft::set_approximate_chunk_size( unsigned preferred_size )
{
    _chunk_size = 1 << (unsigned)floor(log2((float)preferred_size)+0.5);
    return _chunk_size;

//    if (_ok_chunk_sizes.empty())
//        build_performance_statistics(true);

//    std::vector<unsigned>::iterator itr =
//            std::lower_bound( _ok_chunk_sizes.begin(), _ok_chunk_sizes.end(), preferred_size );

//    unsigned N1 = *itr, N2;
//    if (itr == _ok_chunk_sizes.end())
//    {
//        N2 = spo2g( preferred_size - 1 );
//        N1 = lpo2s( preferred_size + 1 );
//    }
//    else if (itr == _ok_chunk_sizes.begin())
//        N2 = N1;
//    else
//        N2 = *--itr;

//    _chunk_size = absdiff(N1, preferred_size) < absdiff(N2, preferred_size) ? N1 : N2;
//    return _chunk_size;
}


unsigned Stft::build_performance_statistics(bool writeOutput, float size_of_test_signal_in_seconds)
{
    _ok_chunk_sizes.clear();
    Tfr::Stft S;
    scoped_ptr<TaskTimer> tt;
    if(writeOutput) tt.reset( new TaskTimer("Building STFT performance statistics for %s", CudaProperties::getCudaDeviceProp().name));
    Signal::pBuffer B = boost::shared_ptr<Signal::Buffer>( new Signal::Buffer( 0, 44100*size_of_test_signal_in_seconds, 44100 ) );
    {
        scoped_ptr<TaskTimer> tt;
        if(writeOutput) tt.reset( new TaskTimer("Filling test buffer with random data (%.1f kB or %.1f s with fs=44100)", B->number_of_samples()*sizeof(float)/1024.f, size_of_test_signal_in_seconds));

        float* p = B->waveform_data()->getCpuMemory();
        for (unsigned i = 0; i < B->number_of_samples(); i++)
            p[i] = rand() / (float)RAND_MAX;
    }

    time_duration fastest_time;
    unsigned fastest_size = 0;
    unsigned ok_size = 0;
    Tfr::pChunk C;
    time_duration latest_time[4];
    unsigned max_base = 3;
    //double base[] = {2,3,5,7};
    double base[] = {2};
    for (unsigned n = 128; n < B->number_of_samples(); n++ )
    {
        unsigned N = -1;
        unsigned selectedBase = 0;

        for (unsigned b=0; b<sizeof(base)/sizeof(base[0]) && b<=max_base; b++)
        {
            double x = ceil(log((double)n)/log(base[b]));
            unsigned N2 = pow(base[b], x);

            if (N2<N)
            {
                selectedBase = b;
                N = N2;
            }
        }

        n = N;

        S._chunk_size = N;

        {
            scoped_ptr<TaskTimer> tt;
            if(writeOutput) tt.reset( new TaskTimer( "n=%u, _chunk_size = %u = %g ^ %g",
                                                     n, S._chunk_size,
                                                     base[selectedBase],
                                                     log2f((float)S._chunk_size)/log2f(base[selectedBase])));

            ptime startTime = microsec_clock::local_time();

            C = S( B );

            if (!C)
                continue;

            time_duration diff = microsec_clock::local_time() - startTime;

            if (0<selectedBase)
                if (diff.total_microseconds() > latest_time[0].total_microseconds()*1.5)
                {
                    max_base = selectedBase-1;
                    if(tt) tt->info("max_base = %u", max_base);
                }

            latest_time[selectedBase] = diff;

            if (diff.total_milliseconds() < fastest_time.total_milliseconds()*1.2)
                ok_size = S._chunk_size;

            if (diff < fastest_time || 0==fastest_size)
            {
                max_base = sizeof(base)/sizeof(base[0]) - 1;
                fastest_time = diff;
                fastest_size = S._chunk_size;
            }

            _ok_chunk_sizes.push_back( S._chunk_size );
        }
        C.reset();

        if (S._chunk_size > B->number_of_samples())
            break;
    }

    if(writeOutput) TaskInfo("Fastest size = %u", fastest_size);
    return fastest_size;
}


StftChunk::
        StftChunk()
            :
            halfs_n(0)
{}


void StftChunk::
        setHalfs( unsigned n )
{
    chunk_offset <<= halfs_n;
    n_valid_samples <<= halfs_n;

    halfs_n = n;

    chunk_offset >>= halfs_n;
    n_valid_samples >>= halfs_n;
}


unsigned StftChunk::
        halfs( )
{
    return halfs_n;
}


unsigned StftChunk::
        nActualScales() const
{
    return Chunk::nScales();
}


unsigned StftChunk::
        nScales() const
{
    return nActualScales() >> halfs_n;
}

} // namespace Tfr
