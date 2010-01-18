#include "cudaUtil.h"
#include <stdio.h>

__global__ void kernel_compute( float* in_waveform_ft, float* out_waveform_ft, cudaExtent numElem, float start, float steplogsize  );
__global__ void kernel_inverse( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem );
__global__ void kernal_clamp( float* in_wt, cudaExtent in_numElem, float* out_clamped_wt, cudaExtent out_numElem, cudaExtent out_offset );

static const char* gLastError = 0;

const char* wtGetError() {
    const char* r = gLastError;
    gLastError = 0;
    return r;
}

void setError(const char* staticErrorMessage) {
    gLastError = staticErrorMessage;
    printf("%s\n", staticErrorMessage);
}

#define TOSTR(x) #x
#define setError(x) setError(TOSTR(__FUNCTION__) ": " x)

void wtCompute( float* in_waveform_ft, float* out_waveform_ft, unsigned sampleRate, float minHz, float maxHz, cudaExtent numElem, cudaStream_t stream )
{
    if(numElem.width%2) {
        setError("Invalid argument, number of floats must be even to compose complex numbers from pairs.");
        return;
    }

    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(numElem.width, block.x), numElem.height*numElem.depth, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    float start = sampleRate/minHz/numElem.width;
    float steplogsize = log(maxHz)-log(minHz);

    kernel_compute<<<grid, block, stream>>>( in_waveform_ft, out_waveform_ft, numElem, start, steplogsize );
}

__global__ void kernel_compute(
        float* in_waveform_ft,
        float* out_waveform_ft,
        cudaExtent numElem, float start, float steplogsize )
{
    // Find period for this thread
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

    // Compute value of analytic FT of wavelet
    const float f0 = 15;
    const float pi = 3.141592654;
    const float two_pi_f0 = 2.0 * pi * f0;
    const float multiplier = 1.8827925275534296252520792527491;

    period *= f0;

    unsigned y = x/2; // compute equal values to multiply the real and imaginary part of in_waveform_ft
    float factor = 2*pi*y*period-two_pi_f0;
    float basic = multiplier * exp(-0.5f*factor*factor);

    // Return
    out_waveform_ft[offset + x] = in_waveform_ft[x]*basic;
}

void wtInverse( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem, cudaStream_t stream  )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(out_numElem.width, block.x), 1, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }
    if(in_numElem.width < 2*out_numElem.width) {
        setError("Invalid argument, complex insignal must be wider than real outsignal.");
        return;
    }

    kernel_inverse<<<grid, block, stream>>>( in_wavelett_ft, in_numElem, out_inverse_waveform, out_numElem );
}

__global__ void kernel_inverse( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem )
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

void wtClamp( float* in_wt, cudaExtent in_numElem, float* out_clamped_wt, cudaExtent out_numElem, cudaExtent out_offset, cudaStream_t stream )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(out_numElem.width, block.x), out_numElem.height, out_numElem.depth );

    if(grid.x>65535) {
        setError("Invalid argument, first dimension of wavelet transform must be less than 65535*256 ~ 16 Mi.");
        return;
    }
    if(grid.y>65535) {
        setError("Invalid argument, number of scales in wavelet transform must be less than 65535.");
        return;
    }
    if(grid.z>1) {
        setError("Invalid argument, out_numElem.depth must be 1.");
        return;
    }

    kernal_clamp<<<grid, block, stream>>>( in_wt, in_numElem, out_clamped_wt, out_numElem, out_offset );
}

__global__ void kernal_clamp( float* in_wt, cudaExtent in_numElem, float* out_clamped_wt, cudaExtent out_numElem, cudaExtent out_offset )
{
    const unsigned
            x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x,
            y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (x>=out_numElem.width )
        return;
    if (y>=out_numElem.height )
        return;
    // sanity checks...
    if (out_offset.width + x >=in_numElem.width)
        return;
    if (out_offset.height + y >=in_numElem.height)
        return;

    // Not coalesced reades for arbitrary out_offset, coalesced writes though
    unsigned i = out_offset.width + x + in_numElem.width*(out_offset.height + y + in_numElem.height*out_offset.depth);
    unsigned o = x + out_numElem.width*y;
    out_clamped_wt[o] = in_wt[i];
}
