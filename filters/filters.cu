#include "cudaUtil.h"
#include <stdio.h>
#include "filters.cu.h"

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#define M_PIf ((float)(M_PI))

__global__ void kernel_remove_disc(float2* in_wavelet, cudaExtent in_numElem, float4 area, bool save_inside );
__global__ void kernel_remove_rect(float2* in_wavelet, cudaExtent in_numElem, float4 area );
__global__ void kernel_move(cudaPitchedPtrType<float2> chunk, float df, float start, float steplogsize, float sample_rate, unsigned sample_offset );
__global__ void kernel_reassign(cudaPitchedPtrType<float2> chunk, float start, float steplogsize, float sample_rate );
__global__ void kernel_tonalize(cudaPitchedPtrType<float2> chunk, float start, float steplogsize, float sample_rate );


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

__global__ void kernel_remove_disc(float2* wavelet, cudaExtent numElem, float4 area, bool save_inside )
{
    unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x,
            fi = __umul24(blockIdx.y,blockDim.y) + threadIdx.y;

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

void moveFilter( cudaPitchedPtrType<float2> chunk, float df, float min_hz, float max_hz, float sample_rate, unsigned sample_offset )
{
    elemSize3_t numElem = chunk.getNumberOfElements();
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.x, block.x), 1, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    //float start = sampleRate/minHz/numElem.width;
    float start = min_hz/2;
    float steplogsize = log(max_hz)-log(min_hz);

    kernel_move<<<grid, block>>>( chunk, df, start, steplogsize, sample_rate, sample_offset );
}

__global__ void kernel_move(cudaPitchedPtrType<float2> chunk, float df, float start, float steplogsize, float sample_rate, unsigned sample_offset )
{
    const unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    unsigned nSamples = chunk.getNumberOfElements().x;
    unsigned nFrequencies = chunk.getNumberOfElements().y;
    if( x >= nSamples )
        return;

    for( unsigned fc = 0<df ? 1:nFrequencies ;
                       0<df ? fc<=nFrequencies : fc>0;
                       0<df?fc++:fc--)
    {
        // fc is a counter that is off by one, it goes [1,nFrequencies] or [nFrequencies,1]
        unsigned fi = fc-1; // fi=write index

        float ri = fi + df; // read index
        float ff_read = ri/(float)nFrequencies;
        float ff_write = fi/(float)nFrequencies;

        //float period = start*exp(-ff*steplogsize); // start = sampleRate/minHz/numElem.width

        float hz_read = start*exp(ff_read*steplogsize); // start = min_hz/2
        float hz_write = start*exp(ff_write*steplogsize); // start = min_hz/2

        elemSize3_t readPos;
        readPos.x = x;
        readPos.y = ri;
        readPos.z = 0;
        float2 w = make_float2(0,0);
        if ( chunk.valid(readPos) ) {
            float2 c = chunk.elem(readPos);

            // compute how many periods have elapsed at x for readPos.y
            float time = (sample_offset+x) / sample_rate; // same time for both read and write
            float read_angle = time * hz_read*2*M_PIf;
            float write_angle = time * hz_write*2*M_PIf;
            float phaseAngle = atan2( c.y, c.x );
            float phase = fmodf(read_angle + phaseAngle + 2*M_PIf, 2*M_PIf);
            float f = write_angle;// + phase;

            float amplitude = sqrt(c.x*c.x + c.y*c.y);
            w.x = cos(f)*amplitude;
            w.y = sin(f)*amplitude;
            w.x = c.x;
            w.y = c.y;
        }

        elemSize3_t writePos;
        writePos.x = x;
        writePos.y = fi;
        writePos.z = 0;

        chunk.e(writePos) = w;
    }
}

void reassignFilter( cudaPitchedPtrType<float2> chunk, float min_hz, float max_hz, float sample_rate )
{
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
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

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

void tonalizeFilter( cudaPitchedPtrType<float2> chunk, float min_hz, float max_hz, float sample_rate )
{
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
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

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
