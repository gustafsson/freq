#include "../misc/inc/cudaUtil.h"
#include <stdio.h>

__global__ void WavelettKernel( float* in_waveform_ft, float* out_waveform_ft, cudaExtent numElem, float start, float steplogsize  );
__global__ void InverseKernel( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem );

void computeWavelettTransform( float* in_waveform_ft, float* out_waveform_ft, unsigned sampleRate, float minHz, float maxHz, cudaExtent numElem);
void inverseWavelettTransform( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem );

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

    const float f0 = 15;
    const float pi = 3.141592654;
    const float two_pi_f0 = 2.0 * pi * f0;
    const float multiplier = 1.8827925275534296252520792527491;

    period *= f0;

    unsigned y = x/2; // compute equal results for the complex and scalar part
    float factor = 2*pi*y*period-two_pi_f0;
    float basic = multiplier * exp(-0.5*factor*factor);

    out_waveform_ft[offset + x] = in_waveform_ft[x]*basic;
}

void inverseWavelettTransform( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem)
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

    InverseKernel<<<grid, block>>>( in_wavelett_ft, in_numElem, out_inverse_waveform, out_numElem );
}

__global__ void InverseKernel(float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem )
{
    const unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (x>=out_numElem.width )
        return;

    float a = 0;
    for (unsigned fi=0; fi<in_numElem.height; fi++)
    {
        // 2*x selects only the real component of the complex transform
        a += in_wavelett_ft[ 2*x + fi*in_numElem.width ];
    }

    out_inverse_waveform[x] = a;
}
