#include "ellipsekernel.h"
#include "resamplecuda.cu.h"
#include "ellipsekerneldef.h"
#include "cudaUtil.h"

#include "cudaglobalstorage.h"

#include <stdio.h>

__global__ void kernel_remove_disc(float2* wavelet, DataStorageSize numElem, Area area, bool save_inside, float fs );

#if 0
void removeDisc( Tfr::ChunkData::Ptr waveletp, Area area, bool save_inside, float fs )
{
    float2* wavelet = (float2*)CudaGlobalStorage::ReadWrite<2>( waveletp ).device_ptr();

    DataStorageSize size = waveletp->size();

    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(size.width, block.x), size.height, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    grid.x *= 2; // To coalesce better, one thread for each float (instead of each float2)
    kernel_remove_disc<<<grid, block>>>( wavelet, size, area, save_inside, fs );
}
#endif

__global__ void kernel_remove_disc(float2* wavelet, DataStorageSize size, Area area, bool save_inside, float fs )
{
    unsigned
            x = blockIdx.x*blockDim.x + threadIdx.x,
            fi = blockIdx.y*blockDim.y + threadIdx.y;

    remove_disc_elem(DataPos(x, fi), wavelet, size, area, save_inside, fs );
}
