#include "cudaUtil.h"
#include <stdio.h>
#include "tfr-filter.cu.h"

__global__ void kernel_remove_disc(float2* in_wavelet, cudaExtent in_numElem, float4 area );
__global__ void kernel_remove_rect(float2* in_wavelet, cudaExtent in_numElem, float4 area );
__global__ void kernel_move(cudaPitchedPtrType<float2> chunk, float df, float start, float steplogsize, float sample_rate );


void removeDisc( float2* wavelet, cudaExtent numElem, float4 area )
{
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

void moveFilter( cudaPitchedPtrType<float2> chunk, float df, float min_hz, float max_hz, float sample_rate )
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

    kernel_move<<<grid, block>>>( chunk, df, start, steplogsize, sample_rate );
}

__global__ void kernel_move(cudaPitchedPtrType<float2> chunk, float df, float start, float steplogsize, float sample_rate )
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
        unsigned fi = fc-1;
        float ri = fi + df;
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
            float time = x / sample_rate; // same time for both read and write
            float read_angle = time * hz_read*2*M_PI;
            float write_angle = time * hz_write*2*M_PI;
            float phaseAngle = atan2( c.y, c.x );
            float phase = fmodf((float)(read_angle + phaseAngle + 2*M_PI), (float)(2*M_PI));
            float f = write_angle + phase;

            float amplitude = sqrt(c.x*c.x + c.y*c.y);
            w.x = cos(f)*amplitude;
            w.y = sin(f)*amplitude;
        }

        elemSize3_t writePos;
        writePos.x = x;
        writePos.y = fi;
        writePos.z = 0;
        chunk.e(writePos) = w;
    }
}
