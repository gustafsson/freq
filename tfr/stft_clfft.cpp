#ifndef USE_OPENCL
#include <stdexcept>
#include "stft.h"
#include "OpenCLContext.h"

#include "cpumemorystorage.h"
#include "complexbuffer.h"
#include "TaskTimer.h"
#include "computationkernel.h"
#include "clfft/clfftkernelbuffer.h"

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
        cl_mem data_in = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*batchSize*sizeof(std::complex<float>), data_i, &fft_error);
        cl_mem data_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*batchSize*sizeof(std::complex<float>), data_o, &fft_error);

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
            out[x] = conj(in[redundantWidth - x]);
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
        cl_mem data_in = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*batchSize*sizeof(std::complex<float>), data_i, &fft_error);
        cl_mem data_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*batchSize*sizeof(std::complex<float>), data_o, &fft_error);

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


void Stft::
        computeWithOoura(DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize actualSize)
{
    float* input = CpuMemoryStorage::ReadOnly<1>( inputbuffer ).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>( transform_data ).ptr();

    // Transform signal
    TIME_STFT TaskTimer tt2("Stft::operator compute");

    Fft ft( false );
    for (unsigned i=0; i < actualSize.height; ++i)
    {
        ft.computeWithOouraR2C(
                CpuMemoryStorage::BorrowPtr<float>( _window_size,
                                                    input + i*_window_size),
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( actualSize.width,
                                                                output + i*actualSize.width)
        );
    }

    TIME_STFT ComputationSynchronize();

    ComputationSynchronize();
    ComputationCheckError();
}




void Stft::
        computeRedundantWithOoura(Tfr::ChunkData::Ptr inputdata, Tfr::ChunkData::Ptr outputdata, DataStorageSize n)
{
    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>(inputdata).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>(outputdata).ptr();

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
}


void Stft::
        inverseWithOoura(Tfr::ChunkData::Ptr inputdata, DataStorage<float>::Ptr outputdata, DataStorageSize n)
{
    const int actualSize = n.width/2 + 1;
    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>( inputdata ).ptr();
    float* output = CpuMemoryStorage::WriteAll<1>( outputdata ).ptr();

    // Transform signal

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

    TIME_STFT ComputationSynchronize();
}


void Stft::
        inverseRedundantWithOoura( Tfr::ChunkData::Ptr inputdata, Tfr::ChunkData::Ptr outputdata, DataStorageSize n )
{
    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>( inputdata ).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>( outputdata ).ptr();

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
}


} // namespace Tfr
#endif // #ifdef USE_OPENCL
