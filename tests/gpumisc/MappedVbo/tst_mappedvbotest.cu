#include "cudaglobalstorage.h"
#include "cudaPitchedPtrType.h"

__global__ void mappedVboTestKernel(
        float* output, elemSize3_t sz)
{
    elemSize3_t  writePos;
    writePos.x = blockIdx.x * 128 + threadIdx.x;
    writePos.y = blockIdx.y * 1 + threadIdx.y;
    if (writePos.x<sz.x && writePos.y < sz.y)
    {
        unsigned o = writePos.x  +  writePos.y * sz.x;
        o = o % 32;
        output[o] = 0;
    }
}


void mappedVboTestCuda( DataStorage<float>::Ptr datap )
{
    cudaPitchedPtrType<float> data(CudaGlobalStorage::ReadOnly<2>( datap ).getCudaPitchedPtr());

    elemSize3_t sz_output = data.getNumberOfElements();
    dim3 block( 128 );
    dim3 grid( int_div_ceil( sz_output.x, block.x ), sz_output.y );
    mappedVboTestKernel<<< grid, block>>>(data.ptr(), sz_output);
}
