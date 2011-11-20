#include <cudaglobalstorage.h>

#include "stftkernel.h"

#include <stdexcept>

__global__ void kernel_stftNormalizeInverse( cudaPitchedPtrType<float> wave, float v );
__global__ void kernel_stftNormalizeInverse( cudaPitchedPtrType<float2> inwave, cudaPitchedPtrType<float> outwave, float v );

void stftNormalizeInverse(
        DataStorage<float>::Ptr wavep,
        unsigned length )
{
    cudaPitchedPtrType<float> wave(CudaGlobalStorage::ReadWrite<1>( wavep ).getCudaPitchedPtr());

    dim3 grid, block;
    unsigned block_size = 256;
    wave.wrapCudaGrid2D( block_size, grid, block );

    if(grid.x>65535) {
        throw std::runtime_error("stftNormalizeInverse: Invalid argument, number of floats in complex signal must be less than 65535*256.");
    }

    kernel_stftNormalizeInverse<<<grid, block, 0>>>( wave, 1.f/length );
}


__global__ void kernel_stftNormalizeInverse( cudaPitchedPtrType<float> wave, float v )
{
    elemSize3_t writePos;
    if( !wave.unwrapCudaGrid( writePos ))
        return;

    wave.e( writePos ) *= v;
}

void stftNormalizeInverse(
        Tfr::ChunkData::Ptr inwavep,
        DataStorage<float>::Ptr outwavep,
        unsigned length )
{
    cudaPitchedPtrType<float2> inwave(CudaGlobalStorage::ReadOnly<1>( inwavep ).getCudaPitchedPtr());
    cudaPitchedPtrType<float> outwave(CudaGlobalStorage::WriteAll<1>( outwavep ).getCudaPitchedPtr());

    dim3 grid, block;
    unsigned block_size = 256;
    inwave.wrapCudaGrid2D( block_size, grid, block );

    if(grid.x>65535) {
        throw std::runtime_error("stftNormalizeInverse: Invalid argument, number of floats in complex signal must be less than 65535*256.");
    }

    kernel_stftNormalizeInverse<<<grid, block, 0>>>( inwave, outwave, 1.f/length );
}


__global__ void kernel_stftNormalizeInverse( cudaPitchedPtrType<float2> inwave, cudaPitchedPtrType<float> outwave, float v )
{
    elemSize3_t writePos;
    if( !inwave.unwrapCudaGrid( writePos ))
        return;

    outwave.e( writePos ) = inwave.e(writePos).x * v;
}
