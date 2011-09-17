#include "stft.h"
#include "complexbuffer.h"
#include "signal/buffersource.h"
#include "wavelet.cu.h"

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
        CufftHandleContext( cudaStream_t stream, unsigned type )
:   _handle(0),
    _stream(stream),
    _type(type)
{
    if (_type == (unsigned)-1)
        _type = CUFFT_C2C;
}


CufftHandleContext::
        ~CufftHandleContext()
{
    destroy();
    _creator_thread.reset();
}


CufftHandleContext::
        CufftHandleContext( const CufftHandleContext& b )
{
    _handle = 0;
    this->_stream = b._stream;
    this->_type = b._type;
}


CufftHandleContext& CufftHandleContext::
        operator=( const CufftHandleContext& b )
{
    destroy();
    this->_stream = b._stream;
    this->_type = b._type;
    return *this;
}


cufftHandle CufftHandleContext::
        operator()( unsigned elems, unsigned batch_size )
{
    if (0 == _handle || _elems != elems || _batch_size != batch_size) {
        this->_elems = elems;
        this->_batch_size = batch_size;
        create();
    } else {
        _creator_thread.throwIfNotSame(__FUNCTION__);
    }
    return _handle;
}


void CufftHandleContext::
        setType(unsigned type)
{
    if (this->_type != type)
    {
        destroy();
        this->_type = type;
    }
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
            (cufftType_t)_type,
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

    cudaExtent input_n = real_buffer->waveform_data()->getNumberOfElements();
    cudaExtent output_n = input_n;

    // The in-signal is padded to a power of 2 (cufft manual suggest a "multiple
    // of 2, 3, 5 or 7" but a power of one is even better) for faster fft calculations
    output_n.width = spo2g( output_n.width - 1 );

    pChunk chunk;

    // TODO choose method based on data size and locality
    if (0 == "ooura")
    {
        ComplexBuffer b( *real_buffer );

        GpuCpuData<float2>* input = b.complex_waveform_data();

        chunk.reset( new StftChunk );
        chunk->transform_data.reset( new GpuCpuData<float2>(
                0,
                output_n,
                GpuCpuVoidData::CpuMemory ));

        computeWithCufft( *input, *chunk->transform_data, -1);
        //computeWithOoura( *input, *chunk->transform_data, -1 );
    }
    else
    {
        if (output_n.width != input_n.width)
            real_buffer = Signal::BufferSource( real_buffer ).readFixedLength( Signal::Interval( real_buffer->sample_offset, output_n.width));

        chunk.reset( new StftChunk(output_n.width) );
        output_n.width = ((StftChunk*)chunk.get())->nScales();
        chunk->transform_data.reset( new GpuCpuData<float2>(
                0,
                output_n,
                GpuCpuVoidData::CpuMemory ));

        computeWithCufftR2C( *real_buffer->waveform_data(), *chunk->transform_data );
    }

    chunk->freqAxis.setLinear( real_buffer->sample_rate, chunk->nScales()/2 );

    chunk->chunk_offset = real_buffer->sample_offset;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = input_n.width;
    chunk->sample_rate = real_buffer->sample_rate / ((StftChunk*)chunk.get())->transformSize();
    chunk->original_sample_rate = real_buffer->sample_rate;
    return chunk;
}


FreqAxis Fft::
        freqAxis( float FS )
{
    FreqAxis fa;
    fa.setLinear( FS, FS/2 );
    return fa;
}


float Fft::
        displayedTimeResolution( float FS, float /*hz*/ )
{
    return 1/FS;
}


Signal::pBuffer Fft::
        backward( pChunk chunk)
{
    Signal::pBuffer r;
    float fs = chunk->original_sample_rate;
    if (((StftChunk*)chunk.get())->window_size == (unsigned)-1)
    {
        unsigned scales = chunk->nScales();

        // original_sample_rate == fs * scales
        ComplexBuffer buffer( 0, scales, fs );

        computeWithCufft(*chunk->transform_data, *buffer.complex_waveform_data(), 1);
        //computeWithOoura(*chunk->transform_data, *buffer.complex_waveform_data(), 1);

        r = buffer.get_real();
    }
    else
    {
        unsigned scales = ((StftChunk*)chunk.get())->window_size;

        r.reset( new Signal::Buffer(0, scales, fs ));

        computeWithCufftC2R(*chunk->transform_data, *r->waveform_data());
    }

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
        ip.resize(2+(1<<(int)(log2f(N+0.5)-1)) + vector_length_test);

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


void Fft::
        computeWithCufftR2C( GpuCpuData<float>& input, GpuCpuData<float2>& output )
{
    cufftReal* i = input.getCudaGlobal().ptr();
    cufftComplex* o = output.getCudaGlobal().ptr();

    BOOST_ASSERT( input.getNumberOfElements().width/2 + 1 == output.getNumberOfElements().width);

    CufftException_SAFE_CALL(cufftExecR2C(
        CufftHandleContext(0, CUFFT_R2C)(input.getNumberOfElements().width, 1),
        i, o));
}


void Fft::
        computeWithCufftC2R( GpuCpuData<float2>& input, GpuCpuData<float>& output )
{
    cufftComplex* i = input.getCudaGlobal().ptr();
    cufftReal* o = output.getCudaGlobal().ptr();

    BOOST_ASSERT( output.getNumberOfElements().width/2 + 1 == input.getNumberOfElements().width);

    CufftException_SAFE_CALL(cufftExecC2R(
        CufftHandleContext(0, CUFFT_C2R)(output.getNumberOfElements().width, 1),
        i, o));
}


/// STFT

std::vector<unsigned> Stft::_ok_chunk_sizes;

Stft::
        Stft( cudaStream_t stream )
:   _stream( stream ),
//    _handle_ctx_c2c(stream, CUFFT_C2C),
//    _handle_ctx_r2c(stream, CUFFT_R2C),
//    _handle_ctx_c2r(stream, CUFFT_C2R),
    _window_size( 1<<11 ),
    _compute_redundant(false)
//    _fft_many( -1 )
{
    compute_redundant( _compute_redundant );
}


Tfr::pChunk Stft::
        operator() (Signal::pBuffer b)
{
    // @see compute_redundant()
    if (compute_redundant())
        return ChunkWithRedundant(b);

    TIME_STFT TaskTimer ti("Stft::operator, _window_size = %d, b = %s", _window_size, b->getInterval().toString().c_str());
    BOOST_ASSERT( 0!=_window_size );

    uint2 actualSize = make_uint2(
            _window_size/2 + 1,
            b->number_of_samples()/_window_size );

    cudaExtent n = make_cudaExtent( actualSize.x * actualSize.y, 1, 1 );

    if (0==n.height) // not enough data
        return Tfr::pChunk();

    if (32768<n.height) // TODO can't handle this
        n.height = 32768;

    Tfr::pChunk chunk( new Tfr::StftChunk(_window_size) );

    chunk->transform_data.reset( new GpuCpuData<float2>(
            0,
            n,
            GpuCpuVoidData::CudaGlobal ));

    chunk->freqAxis = freqAxis( b->sample_rate );
    chunk->chunk_offset = b->sample_offset + _window_size/2;
    chunk->first_valid_sample = 0;
    chunk->sample_rate = b->sample_rate / _window_size;
    ((StftChunk*)chunk.get())->original_sample_rate = b->sample_rate;
    chunk->n_valid_samples = (chunk->nSamples()-1) * _window_size + 1;

    if (0 == b->sample_offset)
    {
        chunk->n_valid_samples += chunk->chunk_offset;
        chunk->chunk_offset = 0;
    }

    cufftReal* input;
    cufftComplex* output;
    if (b->waveform_data()->getMemoryLocation() == GpuCpuVoidData::CpuMemory)
    {
        TIME_STFT TaskTimer tt("fetch input from Cpu to Gpu, %g MB", b->waveform_data()->getSizeInBytes1D()/1024.f/1024.f);
        input = b->waveform_data()->getCudaGlobal().ptr();
        TIME_STFT CudaException_ThreadSynchronize();
    }
    else
    {
        input = b->waveform_data()->getCudaGlobal().ptr();
    }
    output = chunk->transform_data->getCudaGlobal().ptr();

    // Transform signal
    unsigned count = actualSize.y;

    unsigned
            slices = count,
            i = 0;

    size_t free = availableMemoryForSingleAllocation();

    if (slices * _window_size*2*sizeof(cufftComplex) > free)
    {
        slices = free/(_window_size*2*sizeof(cufftComplex));
        slices = std::min(512u, std::min((unsigned)n.height, slices));
    }

    TIME_STFT TaskTimer tt2("Stft::operator compute");

    BOOST_ASSERT( slices > 0 );

    CufftHandleContext
            _handle_ctx_r2c(_stream, CUFFT_R2C);

    while(i < count)
    {
        slices = std::min(slices, count-i);

        try
        {
            CufftException_SAFE_CALL(cufftExecR2C(
                    _handle_ctx_r2c(_window_size, slices),
                    input + i*_window_size,
                    output + i*actualSize.x));

            i += slices;
        } catch (const CufftException& /*x*/) {
            _handle_ctx_r2c(0,0);
            if (slices>1)
                slices/=2;
            else
                throw;
        }
    }

    TIME_STFT CudaException_ThreadSynchronize();

    CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();

    if (false)
    {
        Signal::pBuffer breal = b;
        Signal::pBuffer binv = inverse( chunk );
        float* binv_p = binv->waveform_data()->getCpuMemory();
        float* breal_p = breal->waveform_data()->getCpuMemory();
        Signal::IntervalType breal_length = breal->number_of_samples();
        Signal::IntervalType binv_length = binv->number_of_samples();
        BOOST_ASSERT( breal_length = binv_length );
        float maxd = 0;
        for(Signal::IntervalType i =0; i<breal_length; i++)
        {
            float d = breal_p[i]-binv_p[i];
            if (d*d > maxd)
                maxd = d*d;
        }

        TaskInfo("Difftest %s (value %g)", maxd<1e-8?"passed":"failed", maxd);
    }

    CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();

    return chunk;
}


Tfr::pChunk Stft::
        ChunkWithRedundant(Signal::pBuffer breal)
{
    TIME_STFT TaskTimer ti("Stft::ChunkWithRedundant, _window_size = %d, b = %s", _window_size, breal->getInterval().toString().c_str());

    ComplexBuffer b(*breal);

    BOOST_ASSERT( 0!=_window_size );

    cudaExtent n = make_cudaExtent(
            _window_size,
            b.number_of_samples()/_window_size,
            1 );

    if (0==n.height) // not enough data
        return Tfr::pChunk();

    if (32768<n.height) // TODO can't handle this
        n.height = 32768;

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
    if (slices * _window_size*2*sizeof(cufftComplex) > free)
    {
        slices = free/(_window_size*2*sizeof(cufftComplex));
        slices = std::min(512u, std::min((unsigned)n.height, slices));
    }

    CufftHandleContext
            _handle_ctx_c2c(_stream, CUFFT_C2C);

    while(i < count)
    {
        slices = std::min(slices, count-i);

        try
        {
            CufftException_SAFE_CALL(cufftExecC2C(
                    _handle_ctx_c2c(_window_size, slices),
                    input + i*n.width,
                    output + i*n.width,
                    CUFFT_FORWARD));

            i += slices;
        } catch (const CufftException& /*x*/) {
            _handle_ctx_c2c(0,0);
            if (slices>1)
                slices/=2;
            else
                throw;
        }
    }

    chunk->freqAxis = freqAxis( breal->sample_rate );
    chunk->chunk_offset = b.sample_offset + _window_size/2;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = (chunk->nSamples()-1) * _window_size + 1;
    chunk->sample_rate = b.sample_rate / chunk->nScales();
    ((StftChunk*)chunk.get())->original_sample_rate = breal->sample_rate;

    if (0 == b.sample_offset)
    {
        chunk->n_valid_samples += chunk->chunk_offset;
        chunk->chunk_offset = 0;
    }

    TIME_STFT CudaException_ThreadSynchronize();

    return chunk;
}


Signal::pBuffer Stft::
        inverse( pChunk chunk )
{
    CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();

    if (compute_redundant())
        return inverseWithRedundant( chunk );

    BOOST_ASSERT( chunk->nChannels() == 1 );

    const int chunk_window_size = (int)(chunk->freqAxis.max_frequency_scalar*2 + 0.5f);
    const int actualSize = chunk_window_size/2 + 1;
    int nwindows = chunk->transform_data->getNumberOfElements().width / actualSize;

    //TIME_STFT
            TaskTimer ti("Stft::inverse, chunk_window_size = %d, b = %s", chunk_window_size, chunk->getInterval().toString().c_str());

    int
            firstSample = 0;

    if (chunk->chunk_offset != 0)
        firstSample = chunk->chunk_offset - (UnsignedF)(chunk_window_size/2);

    Signal::pBuffer b(new Signal::Buffer(firstSample, nwindows*chunk_window_size, chunk->original_sample_rate));

    BOOST_ASSERT( 0!= chunk_window_size );

    if (0==nwindows) // not enough data
        return Signal::pBuffer();

    if (32768<nwindows) // TODO can't handle this
        nwindows = 32768;

    const cudaExtent n = make_cudaExtent(
            chunk_window_size,
            nwindows,
            1 );

    CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();

    cufftComplex* input = (cufftComplex*)chunk->transform_data->getCudaGlobal().ptr();
    cufftReal* output = (cufftReal*)b->waveform_data()->getCudaGlobal().ptr();

    CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();

    // Transform signal
    const unsigned count = n.height;

    unsigned
            slices = n.height,
            i = 0;

    // check for available memory
    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);
    free /= 2; // Don't even try to get close to use all memory
    // and never use more than 64 MB
    if (free > 64<<20)
        free = 64<<20;
    if (slices * _window_size*2*sizeof(cufftComplex) > free)
    {
        slices = free/(_window_size*2*sizeof(cufftComplex));
        slices = std::min(512u, std::min((unsigned)n.height, slices));
    }

    CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();

    CufftHandleContext
            _handle_ctx_c2r(_stream, CUFFT_C2R);

    while(i < count)
    {
        slices = std::min(slices, count-i);

        try
        {
            CufftException_SAFE_CALL(cufftExecC2R(
                    _handle_ctx_c2r(_window_size, slices),
                    input + i*actualSize,
                    output + i*n.width));

            CudaException_ThreadSynchronize();
            CudaException_CHECK_ERROR();

            i += slices;
        } catch (const CufftException& /*x*/) {
            _handle_ctx_c2r(0,0);
            if (slices>1)
                slices/=2;
            else
                throw;
        }
    }

    CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();

    stftNormalizeInverse( b->waveform_data()->getCudaGlobal(), n.width );

    CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();

    return b;
}

Signal::pBuffer Stft::
        inverseWithRedundant( pChunk chunk )
{
    BOOST_ASSERT( chunk->nChannels() == 1 );
    int
            chunk_window_size = chunk->nScales(),
            nwindows = chunk->nSamples();

    TIME_STFT TaskTimer ti("Stft::inverse, chunk_window_size = %d, b = %s", chunk_window_size, chunk->getInterval().toString().c_str());

    int
            firstSample = 0;

    if (chunk->chunk_offset != 0)
        firstSample = chunk->chunk_offset - (UnsignedF)(chunk_window_size/2);

    ComplexBuffer b(firstSample, nwindows*chunk_window_size, chunk->original_sample_rate);

    BOOST_ASSERT( 0!= chunk_window_size );

    if (0==nwindows) // not enough data
        return Signal::pBuffer();

    if (32768<nwindows) // TODO can't handle this
        nwindows = 32768;

    cudaExtent n = make_cudaExtent(
            chunk_window_size,
            nwindows,
            1 );

    cufftComplex* input = (cufftComplex*)chunk->transform_data->getCudaGlobal().ptr();
    cufftComplex* output = (cufftComplex*)b.complex_waveform_data()->getCudaGlobal().ptr();

    // Transform signal
    unsigned count = n.height;

    unsigned
            slices = n.height,
            i = 0;

    // check for available memory
    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);
    free /= 2; // Don't even try to get close to use all memory
    // and never use more than 64 MB
    if (free > 64<<20)
        free = 64<<20;
    if (slices * _window_size*2*sizeof(cufftComplex) > free)
    {
        slices = free/(_window_size*2*sizeof(cufftComplex));
        slices = std::min(512u, std::min((unsigned)n.height, slices));
    }

    CufftHandleContext
            _handle_ctx_c2c(_stream, CUFFT_C2C);

    while(i < count)
    {
        slices = std::min(slices, count-i);

        try
        {
            CufftException_SAFE_CALL(cufftExecC2C(
                    _handle_ctx_c2c(_window_size, slices),
                    input + i*n.width,
                    output + i*n.width,
                    CUFFT_INVERSE));

            i += slices;
        } catch (const CufftException& /*x*/) {
            _handle_ctx_c2c(0,0);
            if (slices>1)
                slices/=2;
            else
                throw;
        }
    }

    TIME_STFT CudaException_ThreadSynchronize();

    Signal::pBuffer realinv = b.get_real();
    stftNormalizeInverse( realinv->waveform_data()->getCudaGlobal(), n.width );

    TIME_STFT CudaException_ThreadSynchronize();

    return realinv;
}


FreqAxis Stft::
        freqAxis( float FS )
{
    FreqAxis fa;
    fa.setLinear( FS, _window_size/2 );
    return fa;
}


float Stft::
        displayedTimeResolution( float FS, float /*hz*/ )
{
    return _window_size / FS;
}


//static unsigned absdiff(unsigned a, unsigned b)
//{
//    return a < b ? b - a : a - b;
//}


unsigned powerprod(const unsigned*bases, const unsigned*b, unsigned N)
{
    unsigned v = 1;
    for (unsigned i=0; i<N; i++)
        for (unsigned x=0; x<b[i]; x++)
            v*=bases[i];
    return v;
}

unsigned findGreatestSmaller(const unsigned* bases, unsigned* a, unsigned maxv, unsigned x, unsigned n, unsigned N)
{
    unsigned i = 0;
    while(true)
    {
        a[n] = i;

        unsigned v = powerprod(bases, a, N);
        if (v > x)
            break;

        if (n+1<N)
            maxv = findGreatestSmaller(bases, a, maxv, x, n+1, N);
        else if (v > maxv)
            maxv = v;

        ++i;
    }
    a[n] = 0;

    return maxv;
}

unsigned findSmallestGreater(const unsigned* bases, unsigned* a, unsigned minv, unsigned x, unsigned n, unsigned N)
{
    unsigned i = 0;
    while(true)
    {
        a[n] = i;

        unsigned v = powerprod(bases, a, N);
        if (n+1<N)
            minv = findSmallestGreater(bases, a, minv, x, n+1, N);
        else if (v>=x && (v < minv || minv==0))
            minv = v;

        if (v > x)
            break;

        ++i;
    }
    a[n] = 0;

    return minv;
}

unsigned oksz(unsigned x)
{
    unsigned bases[]={2, 3, 5, 7};
    unsigned a[]={0, 0, 0, 0};
    unsigned gs = findGreatestSmaller(bases, a, 0, x, 0, 4);
    unsigned sg = findSmallestGreater(bases, a, 0, x, 0, 4);
    if (x-gs < sg-x)
        return gs;
    else
        return sg;
}

unsigned Stft::set_approximate_chunk_size( unsigned preferred_size )
{
    //_window_size = 1 << (unsigned)floor(log2f(preferred_size)+0.5);
    _window_size = oksz( preferred_size );
    _window_size = _window_size > 4 ? _window_size : 4;
    return _window_size;

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


void Stft::
        compute_redundant(bool value)
{
    _compute_redundant = value;
    if (_compute_redundant)
    {
        // free unused memory
        //_handle_ctx_c2r(0,0);
        //_handle_ctx_r2c(0,0);
    }
    else
    {
        // free unused memory
        //_handle_ctx_c2c(0,0);
    }
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

        S._window_size = N;

        {
            scoped_ptr<TaskTimer> tt;
            if(writeOutput) tt.reset( new TaskTimer( "n=%u, _chunk_size = %u = %g ^ %g",
                                                     n, S._window_size,
                                                     base[selectedBase],
                                                     log2f((float)S._window_size)/log2f(base[selectedBase])));

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
                ok_size = S._window_size;

            if (diff < fastest_time || 0==fastest_size)
            {
                max_base = sizeof(base)/sizeof(base[0]) - 1;
                fastest_time = diff;
                fastest_size = S._window_size;
            }

            _ok_chunk_sizes.push_back( S._window_size );
        }
        C.reset();

        if (S._window_size > B->number_of_samples())
            break;
    }

    if(writeOutput) TaskInfo("Fastest size = %u", fastest_size);
    return fastest_size;
}


StftChunk::
        StftChunk(unsigned window_size)
            :
            Chunk( Order_column_major ),
            window_size(window_size),
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
    if (window_size == (unsigned)-1)
        return Chunk::nScales();
    return window_size/2 + 1;
}


unsigned StftChunk::
        nSamples() const
{
    if (window_size == (unsigned)-1)
        return Chunk::nSamples();
    return transform_data->getNumberOfElements().width / nActualScales();
}


unsigned StftChunk::
        nScales() const
{
    unsigned s = transformSize();
    if (window_size == (unsigned)-1)
        return s;
    else
        return s/2 + 1;
}


unsigned StftChunk::
        transformSize() const
{
    if (window_size == (unsigned)-1)
        return Chunk::nScales() >> halfs_n;
    else
        return window_size >> halfs_n;
}


} // namespace Tfr
