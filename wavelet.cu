#include "../misc/cudaUtil.h"
#include <stdio.h>

__global__ void WavelettKernel( float* in_waveform_ft, float* out_waveform_ft, cudaExtent numElem, float start, float steplogsize  );
__global__ void InverseKernel( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem, uint4 area );
__global__ void RemoveDiscKernel(float* in_wavelet, cudaExtent in_numElem, uint4 area );

void computeWavelettTransform( float* in_waveform_ft, float* out_waveform_ft, unsigned sampleRate, float minHz, float maxHz, cudaExtent numElem);
void inverseWavelettTransform( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem, uint4 area );
void removeDisc( float* in_wavelet, cudaExtent in_numElem, uint4 area );

void computeWavelettTransform( float* in_waveform_ft, float* out_waveform_ft, unsigned sampleRate, float minHz, float maxHz, cudaExtent numElem)
{
    if(numElem.width%2) {
        printf("Invalid argument, number of floats must be even to compose complex numbers from pairs.");
        return;
    }

    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(numElem.width, block.x), numElem.height*numElem.depth, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    float start = sampleRate/minHz/numElem.width;
    float steplogsize = log(maxHz)-log(minHz);

    WavelettKernel<<<grid, block>>>( in_waveform_ft, out_waveform_ft, numElem, start, steplogsize );
}

__global__ void WavelettKernel(
        float* in_waveform_ft,
        float* out_waveform_ft,
        cudaExtent numElem, float start, float steplogsize )
{
    // Find period for this thread
    // float f = exp(log(minHz) + (fi/(float)nFrequencies())*(log(maxHz)-log(minHz)));
    // return sampleRate/f;

    unsigned nFrequencies = numElem.height;
    unsigned fi = blockIdx.y%nFrequencies;
    float ff = fi/(float)nFrequencies;
    float period = start*exp(-ff*steplogsize);

    // Find offset for this wavelett scale
    unsigned channel = blockIdx.y/nFrequencies; // integer division
    unsigned n = numElem.width;
    unsigned offset = fi*n + channel*n*nFrequencies;

    // Element number
    const unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (x>=numElem.width)
        return;

    const float f0 = .6f + 40*ff*ff*ff;
    const float pi = 3.141592654f;
    const float two_pi_f0 = 2.0f * pi * f0;
    const float multiplier = 1.8827925275534296252520792527491f;

    period *= f0;

    unsigned y = x/2; // compute equal results for the complex and scalar part
    float factor = 4*pi*y*period-two_pi_f0;
    float basic = multiplier * exp(-0.5f*factor*factor);
    out_waveform_ft[offset + x] = in_waveform_ft[x]*basic*f0;
}

void inverseWavelettTransform( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem, uint4 area)
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(out_numElem.width, block.x), 1, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }
    if(in_numElem.width < 2*out_numElem.width) {
        printf("Invalid argument, complex insignal must be wider than real outsignal.");
        return;
    }

    InverseKernel<<<grid, block>>>( in_wavelett_ft, in_numElem, out_inverse_waveform, out_numElem, area );
}

__global__ void InverseKernel(float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem, uint4 area )
{
    const unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (x>=out_numElem.width )
        return;

    float a = 0;

 /* box selection
    if (x>=area.x && x<=area.z) {
        for (unsigned fi=area.y; fi<in_numElem.height && fi<area.w; fi++)
        {
            // 2*x selects only the real component of the complex transform
            a += in_wavelett_ft[ 2*x + fi*in_numElem.width ];
        }
    }*/
/* disc selection */
    for (unsigned fi=0; fi<in_numElem.height; fi++)
    {
        float rx = area.z-(float)area.x;
        float ry = area.w-(float)area.y;
        float dx = x-(float)area.x;
        float dy = fi-(float)area.y;

        if (dx*dx/rx/rx + dy*dy/ry/ry < 1) {
            // 2*x selects only the real component of the complex transform
            a += in_wavelett_ft[ 2*x + fi*in_numElem.width ];
        }
    }

    out_inverse_waveform[x] = a;
}

void removeDisc( float* wavelet, cudaExtent numElem, uint4 area )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(numElem.width, block.x), numElem.height*numElem.depth, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    RemoveDiscKernel<<<grid, block>>>( wavelet, numElem, area );
}

__global__ void RemoveDiscKernel(float* wavelet, cudaExtent numElem, uint4 area )
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
        wavelet[ 2*x + fi*numElem.width ] = 0;
        wavelet[ 2*x + 1 + fi*numElem.width ] = 0;
    }
}
