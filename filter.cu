#include "cudaUtil.h"
#include <stdio.h>
#include "filter.cu.h"

__global__ void kernel_remove_disc(float2* in_wavelet, cudaExtent in_numElem, float4 area );


void removeDisc( float2* wavelet, cudaExtent numElem, float4 area )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), numElem.height*numElem.depth, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_remove_disc<<<grid, block>>>( wavelet, numElem, area );
}

__global__ void kernel_remove_disc(float2* wavelet, cudaExtent numElem, float4 area )
{
    const unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x,
            fi = __umul24(blockIdx.y,blockDim.y) + threadIdx.y;

    if (x>=numElem.width )
        return;

    float rx = area.z-(float)area.x;
    float ry = area.w-(float)area.y;
    float dx = x-(float)area.x;
    float dy = fi-(float)area.y;

    if (dx*dx/rx/rx + dy*dy/ry/ry < 1) {
        wavelet[ x + fi*numElem.width ].x = 0;
        wavelet[ x + fi*numElem.width ].y = 0;
    }
}
