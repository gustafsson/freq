#include "stft.h"
#include <cufft.h>
#include <CudaException.h>
#include <CudaProperties.h>

#include "cudaglobalstorage.h"
#include "complexbuffer.h"

#include "wavelet.cu.h"

//#define TIME_STFT
#define TIME_STFT if(0)

namespace Tfr {

void Fft::
        computeWithCufft( DataStorage<std::complex<float>, 3>::Ptr input, DataStorage<std::complex<float>, 3>::Ptr output, int direction )
{
    TIME_STFT TaskTimer tt("FFt cufft");

    cufftComplex* d = (cufftComplex*)CudaGlobalStorage::WriteAll( output ).device_ptr();
    BOOST_ASSERT( sizeof(cufftComplex) == sizeof(std::complex<float>));
    BOOST_ASSERT( sizeof(cufftComplex) == sizeof(float2));
    cudaMemset( d, 0, output->numberOfBytes() );
    cudaMemcpy( d,
                CudaGlobalStorage::ReadOnly( input ).device_ptr(),
                input->sizeInBytes().width,
                cudaMemcpyDeviceToDevice );

    // TODO require that input and output is of the exact same size. Do padding
    // before calling computeWithCufft. Then use an out-of-place transform
    // instead of copying the entire input first.

    // Transform signal
    CufftHandleContext _fft_single;
    CufftException_SAFE_CALL(cufftExecC2C(
        _fft_single(output->size().width, 1),
        d, d,
        direction==-1?CUFFT_FORWARD:CUFFT_INVERSE));

    TIME_STFT CudaException_ThreadSynchronize();
}


void Fft::
        computeWithCufftR2C( DataStorage<float, 3>::Ptr input, GpuCpuData<float2>& output )
{
    cufftReal* i = CudaGlobalStorage::ReadOnly( input ).device_ptr();
    cufftComplex* o = output.getCudaGlobal().ptr();

    BOOST_ASSERT( input->size().width/2 + 1 == output.getNumberOfElements().width);

    CufftException_SAFE_CALL(cufftExecR2C(
        CufftHandleContext(0, CUFFT_R2C)(input->size().width, 1),
        i, o));
}


void Fft::
        computeWithCufftC2R( GpuCpuData<float2>& input, DataStorage<float, 3>::Ptr output )
{
    cufftComplex* i = input.getCudaGlobal().ptr();
    cufftReal* o = CudaGlobalStorage::WriteAll( output ).device_ptr();

    BOOST_ASSERT( output->size().width/2 + 1 == input.getNumberOfElements().width);

    CufftException_SAFE_CALL(cufftExecC2R(
        CufftHandleContext(0, CUFFT_C2R)(output->size().width, 1),
        i, o));
}


Tfr::pChunk Stft::
        computeWithCufft(Signal::pBuffer b)
{
    uint2 actualSize = make_uint2(
            _window_size/2 + 1,
            b->number_of_samples()/_window_size );

    cudaExtent n = make_cudaExtent( actualSize.x * actualSize.y, 1, 1 );

    if (0==actualSize.y) // not enough data
        return Tfr::pChunk();

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
    if (!b->waveform_data()->DoesStorageHaveValidContent<CudaGlobalStorage>())
    {
        TIME_STFT TaskTimer tt("fetch input from Cpu to Gpu, %g MB", b->waveform_data()->getSizeInBytes1D()/1024.f/1024.f);
        input = CudaGlobalStorage::ReadOnly( b->waveform_data() ).device_ptr();
        TIME_STFT CudaException_ThreadSynchronize();
    }
    else
    {
        input = CudaGlobalStorage::ReadOnly( b->waveform_data() ).device_ptr();
    }
    output = chunk->transform_data->getCudaGlobal().ptr();

    // Transform signal
    unsigned count = actualSize.y;

    unsigned
            slices = count,
            i = 0;

    size_t free = availableMemoryForSingleAllocation();

    unsigned multiple = 0;
    multiple++; // input
    multiple++; // output
    multiple++; // overhead during computaion

    if (slices * _window_size*multiple*sizeof(cufftComplex) > free)
    {
        slices = free/(_window_size*multiple*sizeof(cufftComplex));
        slices = std::min(512u, std::min((unsigned)n.height, slices));

        if (0 == slices) // happens when 'free' is low (migth even be 0)
        {
            // Try with one slice anyways and see take our chances. If it
            // fails an exception will be thrown if either the cufft library
            // or cuda memory allocation fails.
            slices = 1;
        }
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
        computeRedundantWithCufft(Signal::pBuffer breal)
{
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


    cufftComplex* input = (cufftComplex*)CudaGlobalStorage::ReadOnly(b.complex_waveform_data()).device_ptr();
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
        inverseWithCufft(Tfr::pChunk chunk)
{
    CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();
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
    cufftReal* output = (cufftReal*)CudaGlobalStorage::WriteAll( b->waveform_data() ).device_ptr();

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

    stftNormalizeInverse( b->waveform_data(), n.width );

    CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();

    return b;
}


Signal::pBuffer Stft::
        inverseRedundantWithCufft(Tfr::pChunk chunk)
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
    cufftComplex* output = (cufftComplex*)CudaGlobalStorage::WriteAll( b.complex_waveform_data() ).device_ptr();

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
    stftNormalizeInverse( realinv->waveform_data(), n.width );

    TIME_STFT CudaException_ThreadSynchronize();

    return realinv;
}


} // namespace Tfr
