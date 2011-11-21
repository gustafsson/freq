#ifndef USE_OPENCL
#include <stdexcept>
#include "stft.h"
#include "OpenCLContext.h"

#include "cpumemorystorage.h"
#include "complexbuffer.h"
#include "TaskTimer.h"
#include "computationkernel.h"
#include "clfftkernelbuffer.h"

#include "clfft/clFFT.h"

#include "waveletkernel.h"

//#define TIME_STFT
#define TIME_STFT if(0)


namespace Tfr {


void Fft::
        computeWithOoura( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction )
{
    TIME_STFT TaskTimer tt("Fft Ooura");

    unsigned n = input->getNumberOfElements().width;
    unsigned N = output->getNumberOfElements().width;

    if (-1 != direction)
        BOOST_ASSERT( n == N );

    {
        TIME_STFT TaskTimer tt("Computing fft(N=%u, n=%u, direction=%d)", N, n, direction);
        clFFT_Dim3 ndim = { n, 1, 1 };
        int batchSize = 1;
        OpenCLContext *opencl = OpenCLContext::initialize();
        cl_context context = opencl->getContext();
        cl_command_queue queue = opencl->getCommandQueue();
        cl_int fft_error;

		//clFFT_Plan plan = clFFT_CreatePlan(context, ndim, clFFT_1D, clFFT_InterleavedComplexFormat, &fft_error);
        clFFT_Plan plan = CLFFTKernelBuffer::initialize()->getPlan(context, n, &fft_error);
        if(fft_error != 0)
            throw std::runtime_error("Could not create clFFT compute plan.");

        Tfr::ChunkElement* data_i = CpuMemoryStorage::ReadOnly<1>( input ).ptr();
        Tfr::ChunkElement* data_o = CpuMemoryStorage::WriteAll<1>( output ).ptr();

        // Allocate memory for in data
        cl_mem data_in = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, n*batchSize*sizeof(std::complex<float>), data_i, &fft_error);
        cl_mem data_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, n*batchSize*sizeof(std::complex<float>), data_o, &fft_error);

        // Run the fft in OpenCL :)
        fft_error |= clFFT_ExecuteInterleaved(queue, plan, batchSize, (clFFT_Direction)direction, data_in, data_out, 0, NULL, NULL );
        if(fft_error != 0)
            throw std::runtime_error("Bad stuff happened during FFT computation.");

        // Read the memory from OpenCL
        fft_error |= clEnqueueReadBuffer(queue, data_out, CL_TRUE, 0, n*batchSize*sizeof(std::complex<float>), data_o, 0, NULL, NULL);
        if(fft_error != 0)
            throw std::runtime_error("Could not read from OpenCL memory");

        clReleaseMemObject(data_in);
        clReleaseMemObject(data_out);
    }
}


void Fft::
        computeWithOouraR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output )
{
    unsigned denseWidth = output->size().width;
    unsigned redundantWidth = input->size().width;

   BOOST_ASSERT( denseWidth == redundantWidth/2+1 );

    // interleave input to complex data
    Signal::Buffer buffer(0, redundantWidth, 1 );
    //*buffer.waveform_data() = *input;
    new CpuMemoryStorage(buffer.waveform_data().get(), input->getCpuMemory());
    ComplexBuffer complexbuffer( buffer );

    // make room for full output
    Tfr::ChunkData::Ptr redundantOutput( new Tfr::ChunkData( redundantWidth ));

    // compute
    computeWithOoura(complexbuffer.complex_waveform_data(), redundantOutput, FftDirection_Forward);

    // discard redundant output
    {
        Tfr::ChunkElement* in = CpuMemoryStorage::ReadOnly<1>( redundantOutput ).ptr();
        Tfr::ChunkElement* out = CpuMemoryStorage::WriteAll<1>( output ).ptr();
        unsigned x;
        for (x=0; x<denseWidth; ++x)
            out[x] = in[x];
    }
}


void Fft::
        computeWithOouraC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output )
{
    unsigned denseWidth = input->size().width;
    unsigned redundantWidth = output->size().width;

    BOOST_ASSERT( denseWidth == redundantWidth/2+1 );

    Tfr::ChunkData::Ptr redundantInput( new Tfr::ChunkData( redundantWidth, input->size().height ));

    {
        Tfr::ChunkElement* in = CpuMemoryStorage::ReadOnly<1>( input ).ptr();
        Tfr::ChunkElement* out = CpuMemoryStorage::WriteAll<1>( redundantInput ).ptr();
        unsigned x;
        for (x=0; x<denseWidth; ++x)
            out[x] = in[x];
        for (; x<redundantWidth; ++x)
            out[x] = in[redundantWidth - x];
    }

    ComplexBuffer buffer( 0, redundantWidth, 1 );

    computeWithOoura(redundantInput, buffer.complex_waveform_data(), FftDirection_Backward);

    *output = *buffer.get_real()->waveform_data();
}


void Stft::
        computeWithOoura( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction )
{
    TaskTimer tt("Stft::computeWithOoura( matrix[%d, %d], %s )",
                 input->size().width,
                 input->size().height,
                 direction==FftDirection_Forward?"forward":"backward");

    Tfr::ChunkElement* data_i = CpuMemoryStorage::ReadOnly<1>( input ).ptr();
    Tfr::ChunkElement* data_o = CpuMemoryStorage::WriteAll<1>( output ).ptr();

    BOOST_ASSERT( output->numberOfBytes() == input->numberOfBytes() );

    const int count = input->numberOfElements()/_window_size;

    cl_int n = _window_size;
    clFFT_Dim3 ndim = { n, 1, 1 };
    int batchSize = count;
    OpenCLContext *opencl = OpenCLContext::initialize();
    cl_context context = opencl->getContext();
    cl_command_queue queue = opencl->getCommandQueue();
    cl_int fft_error;

    clFFT_Plan plan;
    {
        TaskTimer tt("Creating a OpenCL FFT compute plan!");
		//plan = clFFT_CreatePlan(context, ndim, clFFT_1D, clFFT_InterleavedComplexFormat, &fft_error);
        plan = CLFFTKernelBuffer::initialize()->getPlan(context, n, &fft_error);
    }
    {
        TaskTimer tt("Calculating batches");
        if(fft_error != CL_SUCCESS)
            throw std::runtime_error("Could not create clFFT compute plan.");

        // Allocate memory for in data
        cl_mem data_in = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, n*batchSize*sizeof(std::complex<float>), data_i, &fft_error);
        cl_mem data_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, n*batchSize*sizeof(std::complex<float>), data_o, &fft_error);

        // Run the fft in OpenCL :)
        fft_error |= clFFT_ExecuteInterleaved(queue, plan, batchSize, (clFFT_Direction)direction, data_in, data_out, 0, NULL, NULL );
        if(fft_error != CL_SUCCESS)
            throw std::runtime_error("Bad stuff happened during FFT computation.");

        // Read the memory from OpenCL
        fft_error |= clEnqueueReadBuffer(queue, data_out, CL_TRUE, 0, n*batchSize*sizeof(std::complex<float>), data_o, 0, NULL, NULL);
        if(fft_error != CL_SUCCESS)
            throw std::runtime_error("Could not read from OpenCL memory");

        clReleaseMemObject(data_in);
        clReleaseMemObject(data_out);
    }
}


Tfr::pChunk Stft::
        computeWithOoura(Signal::pBuffer b)
{
    DataStorageSize actualSize(
            _window_size,
            b->number_of_samples()/_window_size );

    DataStorageSize n = actualSize.width * actualSize.height;

    if (0==actualSize.height) // not enough data
        return Tfr::pChunk();

    Tfr::pChunk chunk( new Tfr::StftChunk(_window_size) );

    chunk->transform_data.reset( new Tfr::ChunkData( n ));

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

    ComplexBuffer complex(*b);
    computeWithOoura(complex.complex_waveform_data(), chunk->transform_data, Tfr::FftDirection_Forward);

    TIME_STFT ComputationSynchronize();

    ComputationSynchronize();
    ComputationCheckError();

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

    ComputationSynchronize();
    ComputationCheckError();

    return chunk;
}


Tfr::pChunk Stft::
        computeRedundantWithOoura(Signal::pBuffer breal)
{
    ComplexBuffer b(*breal);

    BOOST_ASSERT( 0!=_window_size );

    DataStorageSize n(
            _window_size,
            b.number_of_samples()/_window_size );

    if (0==n.height) // not enough data
        return Tfr::pChunk();

    if (32768<n.height) // TODO can't handle this
        n.height = 32768;

    Tfr::pChunk chunk( new Tfr::StftChunk() );

    chunk->transform_data.reset( new ChunkData( n ));


    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>(b.complex_waveform_data()).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>(chunk->transform_data).ptr();

    // Transform signal

    Fft ft( true );

    for (unsigned i=0; i < n.height; ++i)
    {
        ft.computeWithOoura(
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                                input + i*n.width),
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                                output + i*n.width),
                FftDirection_Forward
        );
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

    TIME_STFT ComputationSynchronize();

    return chunk;
}


Signal::pBuffer Stft::
        inverseWithOoura(Tfr::pChunk chunk)
{
    ComputationSynchronize();
    ComputationCheckError();
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

    const DataStorageSize n(
            chunk_window_size,
            nwindows );

    ComputationSynchronize();
    ComputationCheckError();

    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>( chunk->transform_data ).ptr();
    float* output = CpuMemoryStorage::WriteAll<1>( b->waveform_data() ).ptr();

    ComputationSynchronize();
    ComputationCheckError();

    // Transform signal

    ComputationSynchronize();
    ComputationCheckError();

    Fft ft(false);

    for (unsigned i=0; i < n.height; ++i)
    {
        ft.computeWithOouraC2R(
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( actualSize,
                                                                input + i*actualSize),
                CpuMemoryStorage::BorrowPtr<float>( n.width,
                                                    output + i*n.width)
        );
    }

    ComputationSynchronize();
    ComputationCheckError();

    stftNormalizeInverse( b->waveform_data(), n.width );

    ComputationSynchronize();
    ComputationCheckError();

    return b;
}


Signal::pBuffer Stft::
        inverseRedundantWithOoura(Tfr::pChunk chunk)
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

    DataStorageSize n(
            chunk_window_size,
            nwindows );

    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>( chunk->transform_data ).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>( b.complex_waveform_data() ).ptr();

    // Transform signal

    Fft ft(true);

    for (unsigned i=0; i < n.height; ++i)
    {
        ft.computeWithOoura(
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                                input + i*n.width),
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                                output + i*n.width),
                FftDirection_Backward
        );
    }

    TIME_STFT ComputationSynchronize();

    Signal::pBuffer realinv = b.get_real();
    stftNormalizeInverse( realinv->waveform_data(), n.width );

    TIME_STFT ComputationSynchronize();

    return realinv;
}


} // namespace Tfr
#endif // #ifdef USE_OPENCL
