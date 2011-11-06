#include "resamplecuda.cu.h"
#include "cuda_vector_types_op.h"
#include "rectanglekerneldef.h"

// gpumisc
#include <cudaUtil.h>

// stdc
#include <stdio.h>

__global__ void kernel_remove_rect(float2* in_wavelet, DataStorageSize in_numElem, Area area, float save_inside );


void removeRect( Tfr::ChunkData::Ptr waveletp, Area area, bool save_inside )
{
    float2* wavelet = (float2*)CudaGlobalStorage::ReadWrite<2>( waveletp ).device_ptr();
    DataStorageSize size = waveletp->size();

    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(size.width, block.x), size.height, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_remove_rect<<<grid, block>>>( wavelet, size, area, save_inside );
}


__global__ void kernel_remove_rect(float2* wavelet, DataStorageSize numElem, Area area, float save_inside )
{
    const DataPos p(
            blockIdx.x*blockDim.x + threadIdx.x,
            blockIdx.y*blockDim.y + threadIdx.y);

    remove_rect_elem(p, wavelet, numElem, area, save_inside );
}
