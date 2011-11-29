#ifdef USE_CUDA
#include "stft.h"

#include <cufft.h>
#include "cudaMemsetFix.cu.h"

#include <CudaException.h>
#include <CudaProperties.h>
#include "cudaglobalstorage.h"
#include "complexbuffer.h"
#include "TaskTimer.h"
#include "cuffthandlecontext.h"

//#define TIME_STFT
#define TIME_STFT if(0)


namespace Tfr {


void Fft::
        computeWithCufft( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction )
{
    TIME_STFT TaskTimer tt("FFt cufft");

    cufftComplex* d = (cufftComplex*)CudaGlobalStorage::WriteAll<1>( output ).device_ptr();
    BOOST_ASSERT( sizeof(cufftComplex) == sizeof(std::complex<float>));
    BOOST_ASSERT( sizeof(cufftComplex) == sizeof(float2));
    unsigned inN = input->sizeInBytes().width,
             outN = output->numberOfBytes();
    if (inN<outN)
        cudaMemsetFix( (char*)d, outN-inN );
    cudaMemcpy( d,
                CudaGlobalStorage::ReadOnly<1>( input ).device_ptr(),
                inN,
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
        computeWithCufftR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output )
{
    cufftReal* i = CudaGlobalStorage::ReadOnly<1>( input ).device_ptr();
    cufftComplex* o = (cufftComplex*)CudaGlobalStorage::WriteAll<1>( output ).device_ptr();

    BOOST_ASSERT( input->size().width/2 + 1 == output->size().width);

    CufftException_SAFE_CALL(cufftExecR2C(
        CufftHandleContext(0, CUFFT_R2C)(input->size().width, 1),
        i, o));
}


void Fft::
        computeWithCufftC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output )
{
    cufftComplex* i = (cufftComplex*)CudaGlobalStorage::ReadOnly<1>( input ).device_ptr();
    cufftReal* o = CudaGlobalStorage::WriteAll<1>( output ).device_ptr();

    BOOST_ASSERT( output->size().width/2 + 1 == input->size().width);

    CufftException_SAFE_CALL(cufftExecC2R(
        CufftHandleContext(0, CUFFT_C2R)(output->size().width, 1),
        i, o));
}


//void Stft::
//        canonicalComputeWithCufft( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, DataStorageSize n, FftDirection direction )
//{
//    cufftComplex* i = (cufftComplex*)CudaGlobalStorage::ReadOnly<1>( input ).device_ptr();
//    cufftComplex* o = (cufftComplex*)CudaGlobalStorage::WriteAll<1>( output ).device_ptr();

//    BOOST_ASSERT( output->numberOfBytes() == input->numberOfBytes() );

//    CufftException_SAFE_CALL(cufftExecC2C(
//        CufftHandleContext(0, CUFFT_C2C)(n.width, n.height),
//        i, o, direction==FftDirection_Forward?CUFFT_FORWARD:CUFFT_BACKWARD));
//}



void Stft::
        computeWithCufft(Tfr::ChunkData::Ptr inputdata, Tfr::ChunkData::Ptr outputdata, DataStorageSize n, FftDirection direction)
{
    cufftComplex* input = (cufftComplex*)CudaGlobalStorage::ReadOnly<1>(inputdata).device_ptr();
    cufftComplex* output = (cufftComplex*)CudaGlobalStorage::WriteAll<1>(outputdata).device_ptr();

    // Transform signal
    const unsigned count = n.height;

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
            _handle_ctx_c2c(0, CUFFT_C2C);

    while(i < count)
    {
        slices = std::min(slices, count-i);

        try
        {
            CufftException_SAFE_CALL(cufftExecC2C(
                    _handle_ctx_c2c(_window_size, slices),
                    input + i*n.width,
                    output + i*n.width,
                    direction==FftDirection_Forward?CUFFT_FORWARD:CUFFT_INVERSE));

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
}


void Stft::
        computeWithCufft(DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize actualSize)
{
    cufftReal* input;
    cufftComplex* output;

    if (!inputbuffer->HasValidContent<CudaGlobalStorage>())
    {
        TIME_STFT TaskTimer tt("fetch input from Cpu to Gpu, %g MB", inputbuffer->getSizeInBytes1D()/1024.f/1024.f);
        input = CudaGlobalStorage::ReadOnly<1>( inputbuffer ).device_ptr();
        TIME_STFT CudaException_ThreadSynchronize();
    }
    else
    {
        input = CudaGlobalStorage::ReadOnly<1>( inputbuffer ).device_ptr();
    }
    output = (cufftComplex*)CudaGlobalStorage::WriteAll<1>( transform_data ).device_ptr();

    // Transform signal
    const unsigned count = actualSize.height;

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
        slices = std::min(512u, std::min((unsigned)actualSize.height, slices));

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
            _handle_ctx_r2c(0, CUFFT_R2C);

    while(i < count)
    {
        slices = std::min(slices, count-i);

        try
        {
            CufftException_SAFE_CALL(cufftExecR2C(
                    _handle_ctx_r2c(_window_size, slices),
                    input + i*_window_size,
                    output + i*actualSize.width));

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
}


void Stft::
        inverseWithCufft( Tfr::ChunkData::Ptr inputdata, DataStorage<float>::Ptr outputdata, DataStorageSize n )
{
    const int actualSize = n.width/2 + 1;
    cufftComplex* input = (cufftComplex*)CudaGlobalStorage::ReadOnly<1>( inputdata ).device_ptr();
    cufftReal* output = (cufftReal*)CudaGlobalStorage::WriteAll<1>( outputdata ).device_ptr();

    TIME_STFT CudaException_ThreadSynchronize();

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

    TIME_STFT CudaException_ThreadSynchronize();

    CufftHandleContext
            _handle_ctx_c2r(0, CUFFT_C2R);

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

    TIME_STFT CudaException_ThreadSynchronize();
}


} // namespace Tfr
#endif // #ifdef USE_CUDA
