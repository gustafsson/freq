#include "cudaUtil.h"
#include <stdio.h>
#include "filter.cu.h"

__global__ void kernel_remove_disc(float2* in_wavelet, cudaExtent in_numElem, float4 area );
__global__ void kernel_remove_rect(float2* in_wavelet, cudaExtent in_numElem, float4 area );


void removeDisc( float2* wavelet, cudaExtent numElem, float4 area )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), numElem.height, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_remove_disc<<<grid, block>>>( wavelet, numElem, area );
}

void removeRect( float2* wavelet, cudaExtent numElem, float4 area )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), numElem.height, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_remove_rect<<<grid, block>>>( wavelet, numElem, area );
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
    float dx = x+.5f-(float)area.x;
    float dy = fi+1.5f-(float)area.y;

    float f = dx*dx/rx/rx + dy*dy/ry/ry;
    float g = dx*dx/(rx+1)/(rx+1) + dy*dy/(ry+1)/(ry+1);
    if (f < 1) {
        f = 0;
    } else if (g<1) {
      f = (1 - 1/f) / (1/g - 1/f);
    } else {
      f = 1;
    }

    if (f < 1) {
        f*=f;
        f*=f;
        wavelet[ x + fi*numElem.width ].x *= f;
        wavelet[ x + fi*numElem.width ].y *= f;
    }
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