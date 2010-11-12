#include "ellipse.cu.h"

// todo remove
#include "cudaUtil.h"

#include <stdio.h>

__global__ void kernel_remove_disc(float2* in_wavelet, cudaExtent in_numElem, float4 area, bool save_inside );

void removeDisc( float2* wavelet, cudaExtent numElem, float4 area, bool save_inside )
{
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), numElem.height, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    grid.x *= 2; // To coalesce better, one thread for each float (instead of each float2)
    kernel_remove_disc<<<grid, block>>>( wavelet, numElem, area, save_inside );
}

__global__ void kernel_remove_disc(float2* wavelet, cudaExtent numElem, float4 area, bool save_inside )
{
    unsigned
            x = blockIdx.x*blockDim.x + threadIdx.x,
            fi = blockIdx.y*blockDim.y + threadIdx.y;

    bool complex = x%2;
    x/=2;

    if (x>=numElem.width )
        return;

    float rx = fabs(area.z - area.x);
    float ry = fabs(area.w - area.y);
    float dx = fabs(x+.5f - area.x);
    float dy = fabs(fi-.5f - area.y);

    float g = dx*dx/rx/rx + dy*dy/ry/ry;
    rx = max(0.f, rx-1000);
    ry = max(0.f, ry-2);
    float f = dx*dx/rx/rx + dy*dy/ry/ry;
    if (f < 1) {
        f = 0;
    } else if (g<1) {
      f = (1 - 1/f) / (1/g - 1/f);
    } else {
      f = 1;
    }

    if (save_inside)
        f = 1-f;

    if (f < 1) {
        //f*=(1-f);
        //f*=(1-f);

        if (f != 0)
            f *= ((float*)wavelet)[ 2*x + complex + fi*2*numElem.width ];

        ((float*)wavelet)[ 2*x + complex + fi*2*numElem.width ] = f;
    }
}
