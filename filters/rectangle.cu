#include "rectangle.cu.h"

// gpumisc
#include <cudaUtil.h>

// stdc
#include <stdio.h>

__global__ void kernel_remove_rect(float2* in_wavelet, cudaExtent in_numElem, float4 area );


void removeRect( float2* wavelet, cudaExtent numElem, float4 area )
{
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), numElem.height, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_remove_rect<<<grid, block>>>( wavelet, numElem, area );
}

__global__ void kernel_remove_rect(float2* wavelet, cudaExtent numElem, float4 area )
{
    const unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x,
            fi = __umul24(blockIdx.y,blockDim.y) + threadIdx.y;

    if (x>=numElem.width )
        return;
    float dx = area.x;
    float dy = area.y;
    float dh = area.z - area.x;
    float dw = area.w - area.y;
    float f;

    if(x > dx - dh && x < dx + dh && fi > dy - dw && fi < dy + dw)
    {
        f = 0;
    }
    else
    {
        f = 1;
    }
    wavelet[ x + fi*numElem.width ].x *= f;
    wavelet[ x + fi*numElem.width ].y *= f;
}
