#include "rectangle.cu.h"

// gpumisc
#include <cudaUtil.h>
#include "cudaglobalstorage.h"

// stdc
#include <stdio.h>

__global__ void kernel_remove_rect(float2* in_wavelet, cudaExtent in_numElem, float4 area, float save_inside );


void removeRect( Tfr::ChunkData::Ptr waveletp, Area a, bool save_inside )
{
    float2* wavelet = (float2*)CudaGlobalStorage::ReadWrite<2>( waveletp ).device_ptr();
    DataStorageSize size = waveletp->size();
    cudaExtent extent = make_cudaExtent(size.width, size.height, size.depth);

    float4 area = { a.x1, a.y1, a.x2, a.y2 };

    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(size.width, block.x), size.height, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_remove_rect<<<grid, block>>>( wavelet, extent, area, save_inside );
}

__global__ void kernel_remove_rect(float2* wavelet, cudaExtent numElem, float4 area, float save_inside )
{
    const unsigned
            x = blockIdx.x*blockDim.x + threadIdx.x,
            fi = blockIdx.y*blockDim.y + threadIdx.y;

    if (x>=numElem.width )
        return;

    float f;

    if(x >= area.x && x <= area.z && fi >= area.y && fi <= area.w)
    {
        f = save_inside;
    }
    else
    {
        f = !save_inside;
    }

    if (f == 0.f)
    {
        float dx = min(fabsf(x-area.x), fabsf(x-area.z));
        float dy = min(fabsf(fi-area.y), fabsf(fi-area.y));
        float f = 1.f - min(dy*(1/1.f), dx*(1/4.f));
        if (f < 0.f)
            f = 0.f;
    }

    wavelet[ x + fi*numElem.width ].x *= f;
    wavelet[ x + fi*numElem.width ].y *= f;
}
