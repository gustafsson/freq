#include "ellipse.cu.h"

#include <cuda_runtime.h>

#include "cudaUtil.h"

#include "cudaglobalstorage.h"

#include <stdio.h>

__global__ void kernel_remove_disc(float2* in_wavelet, cudaExtent in_numElem, float4 area, bool save_inside, float fs );

void removeDisc( Tfr::ChunkData::Ptr waveletp, Area a, bool save_inside, float fs )
{
    float2* wavelet = (float2*)CudaGlobalStorage::ReadWrite<2>( waveletp ).device_ptr();

    DataStorageSize size = waveletp->size();
    cudaExtent extent = make_cudaExtent(size.width, size.height, size.depth);

    float4 area = make_float4( a.x1, a.y1, a.x2, a.y2 );

    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(size.width, block.x), size.height, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    grid.x *= 2; // To coalesce better, one thread for each float (instead of each float2)
    kernel_remove_disc<<<grid, block>>>( wavelet, extent, area, save_inside, fs );
}

__global__ void kernel_remove_disc(float2* wavelet, cudaExtent numElem, float4 area, bool save_inside, float fs )
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
    //float dx = fabs(x+.5f - area.x);
    //float dy = fabs(fi-.5f - area.y);
    float dx = fabs(x - area.x);
    float dy = fabs(fi - area.y);

    float ax = 0.03f*fs; // TODO this should be wavelet_time_support_samples( fs, hz ) = k*2^((b+fi)/scales_per_octave)
    float ay = 1.5f; // only round in time?

    // round corners
    float f = dx*dx/rx/rx + dy*dy/ry/ry;

    rx += ax;
    ry += ay;

    float g = dx*dx/rx/rx + dy*dy/ry/ry;
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
        f = 3*f*f - 2*f*f*f;
        //f*=(1-f);
        //f*=(1-f);

        if (f != 0)
            f *= ((float*)wavelet)[ 2*x + complex + fi*2*numElem.width ];

        ((float*)wavelet)[ 2*x + complex + fi*2*numElem.width ] = f;
    }
}
