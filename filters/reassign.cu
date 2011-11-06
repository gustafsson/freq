#include "cudaglobalstorage.h"

#include "cudaUtil.h"
#include <stdio.h>
#include "reassign.cu.h"

#include <cudaPitchedPtrType.h>

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#define M_PIf ((float)(M_PI))


__global__ void kernel_reassign(cudaPitchedPtrType<float2> chunk, float start, float steplogsize, float sample_rate );
__global__ void kernel_tonalize(cudaPitchedPtrType<float2> chunk, float start, float steplogsize, float sample_rate );

void reassignFilter( Tfr::ChunkData::Ptr chunkp, float min_hz, float max_hz, float sample_rate )
{
    cudaPitchedPtrType<float2> chunk(CudaGlobalStorage::ReadWrite<2>( chunkp ).getCudaPitchedPtr());

    elemSize3_t numElem = chunk.getNumberOfElements();
    dim3 block(32,1,1);
    dim3 grid( int_div_ceil(numElem.x, block.x), 1, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    float start = min_hz;
    float steplogsize = log(max_hz)-log(min_hz);

    kernel_reassign<<<grid, block>>>( chunk, start, steplogsize, sample_rate );
}


__global__ void kernel_reassign(cudaPitchedPtrType<float2> chunk, float start, float steplogsize, float sample_rate )
{
    unsigned
            x = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned nSamples = chunk.getNumberOfElements().x;
    unsigned nFrequencies = chunk.getNumberOfElements().y;
    int s = threadIdx.x%2?1:-1;
    s*=1;
    if( x >= nSamples-abs(s) )
        return;

    for( unsigned fi = 0; fi < nFrequencies; fi ++)
    {
        float2 base = chunk.elem( make_elemSize3_t(x,fi,0) );
        float2 base2 = chunk.elem( make_elemSize3_t(x+s,fi,0) );
        float da = atan2(base2.y, base2.x) - atan2(base.y, base.x);

        if (da>M_PIf) da -= 2*M_PIf;
        if (da<-M_PIf) da += 2*M_PIf;

        float hz_root = abs(da * (sample_rate/s) / (2*M_PIf));

        __syncthreads();

        unsigned root_row = (unsigned)(log(hz_root/start)/steplogsize*nFrequencies);
        if (root_row != fi)
        {
            float2 target = chunk.elem( make_elemSize3_t(x,root_row,0) );
            target.x += base.x;
            target.y += base.y;
            chunk.elem( make_elemSize3_t(x,root_row,0) ) = target;
            chunk.elem( make_elemSize3_t(x,fi,0) ) = make_float2(0,0);
        }
    }
}

void tonalizeFilter( Tfr::ChunkData::Ptr chunkp, float min_hz, float max_hz, float sample_rate )
{
    cudaPitchedPtrType<float2> chunk(CudaGlobalStorage::ReadWrite<2>( chunkp ).getCudaPitchedPtr());

    elemSize3_t numElem = chunk.getNumberOfElements();
    dim3 block(32,1,1);
    dim3 grid( int_div_ceil(numElem.x, block.x), 1, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    float start = min_hz;
    float steplogsize = log(max_hz)-log(min_hz);

    kernel_tonalize<<<grid, block>>>( chunk, start, steplogsize, sample_rate );
}

__global__ void kernel_tonalize(cudaPitchedPtrType<float2> chunk, float start, float steplogsize, float sample_rate )
{
    unsigned
            x = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned nSamples = chunk.getNumberOfElements().x;
    unsigned nFrequencies = chunk.getNumberOfElements().y;
    int s = threadIdx.x%2?1:-1;
    s*=1;
    if( x >= nSamples-abs(s) )
        return;

    for( unsigned fi = 0; fi < nFrequencies; fi ++)
    {
        float ff_root = fi/(float)nFrequencies;
        float hz_root = start*exp(ff_root*steplogsize);

        float2 base = chunk.elem( make_elemSize3_t(x,fi,0) );
        //float2 base2 = chunk.elem( make_elemSize3_t(x+s,fi,0) );
        //float da = atan2(base2.y, base2.x) - atan2(base.y, base.x);

        //if (da>M_PIf) da -= 2*M_PIf;
        //if (da<-M_PIf) da += 2*M_PIf;

        //float hz_root = abs(da * (sample_rate/s) / (2*M_PIf));
        //__syncthreads();

        unsigned root_row = (unsigned)(log(hz_root/start)/steplogsize*nFrequencies);
        //if (root_row == fi)
        {

            float global_max = sqrt(base.x*base.x + base.y*base.y);

            // Loop through overtones
            for( unsigned t = 2; t < 20; t ++)
            {
                float hz_overtone = hz_root*t;

                // Expect a gaussian around this overtone, width=2 hz;
                float hz_start = hz_overtone - 2;
                float hz_end = hz_overtone + 2;

                float read_start = log(hz_start/start)/steplogsize*nFrequencies;
                float read_end = log(hz_end/start)/steplogsize*nFrequencies;

                if (read_start>nFrequencies-1)
                    t=20;
                else
                {
                    float2 sum = {0,0};

                    read_start = floor(read_start);
                    read_end = min(nFrequencies-1, (unsigned)ceil(read_end));
                    float read_middle = 0.5f*read_start + 0.5f*read_end;

                    // Find max gauss
                    float maxg = global_max;
                    for( float r = read_start; r<=read_end; r++)
                    {
                        float2 c = chunk.elem( make_elemSize3_t(x,r,0) );
                        float A = sqrt(c.x*c.x + c.y*c.y);

                        float g = exp(-0.2f*(r-read_middle)*(r-read_middle));

                        if (A<maxg*g)
                            maxg=A/g;
                    }

                    // Remove gauss
                    for( float r = read_start; r<=read_end; r++)
                    {
                        float2 c = chunk.elem( make_elemSize3_t(x,r,0) );
                        float A = sqrt(c.x*c.x + c.y*c.y);

                        float g = exp(-0.2f*(r-read_middle)*(r-read_middle));

                        float f = maxg*g / A;
                        f = min(f, 1.0f);

                        float2 R = {c.x*f, c.y*f};
                        c.x -= R.x;
                        c.y -= R.y;

                        sum.x += R.x;
                        sum.y += R.y;

                        chunk.elem( make_elemSize3_t(x,r,0) ) = c;
                    }
                    base.x += sum.x;
                    base.y += sum.y;
//                    global_max = max(global_max, sqrt(base.x*base.x + base.y*base.y));
                }

            } // next overtone

            chunk.elem( make_elemSize3_t(x,fi,0) ) = base;
        }
    } // next root tone
}
