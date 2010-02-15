#include "cudaUtil.h"
#include <stdio.h>

__global__ void kernel_compute( float* in_waveform_ft, float* out_wavelet_ft, cudaExtent numElem, float start, float steplogsize  );
__global__ void kernel_inverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem );
__global__ void kernel_clamp( float* in_wt, cudaExtent in_numElem, size_t in_offset, size_t last_sample, float* out_clamped_wt, cudaExtent out_numElem );

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

void wtCompute( float2* in_waveform_ft, float2* out_wavelet_ft, unsigned sampleRate, float minHz, float maxHz, cudaExtent numElem, cudaStream_t stream )
{
    // in this scope, work on arrays of float* instead of float2* to coalesce better
    numElem.width *= 2;

    float start = sampleRate/minHz/numElem.width;
    float steplogsize = log(maxHz)-log(minHz);

    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(numElem.width, block.x), numElem.height*numElem.depth, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_compute<<<grid, block, stream>>>( (float*)in_waveform_ft, (float*)out_wavelet_ft, numElem, start, steplogsize );
}

__global__ void kernel_compute(
        float* in_waveform_ft,
        float* out_wavelet_ft,
        cudaExtent numElem, float start, float steplogsize )
{
    // Find period for this thread
    unsigned nFrequencies = numElem.height;
    unsigned fi = blockIdx.y%nFrequencies;
    float ff = fi/(float)nFrequencies;
    float period = start*exp(-ff*steplogsize);

    // Find offset for this wavelet scale
    unsigned channel = blockIdx.y/nFrequencies; // integer division
    unsigned n = numElem.width;
    unsigned offset = fi*n + channel*n*nFrequencies;

    // Element number
    const unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (x>=numElem.width)
        return;

    // Compute value of analytic FT of wavelet
    const float f0 = .6f + 40*ff*ff*ff;
    const float pi = 3.141592654f;
    const float two_pi_f0 = 2.0f * pi * f0;
    const float multiplier = 1.8827925275534296252520792527491f;

    period *= f0;

    unsigned y = x/2; // compute equal results for the complex and scalar part
    float factor = 4*pi*y*period-two_pi_f0;
    float basic = multiplier * exp(-0.5f*factor*factor);

    float jibberish_normalization = 2.3406;
    float cufft_normalize = 1.f/numElem.width;

    out_wavelet_ft[offset + x] = jibberish_normalization*cufft_normalize*basic*f0*in_waveform_ft[x];
}

void wtInverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, cudaStream_t stream  )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(numElem.width, block.x), 1, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_inverse<<<grid, block, stream>>>( in_wavelet, out_inverse_waveform, numElem );
}

__global__ void kernel_inverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem )
{
    const unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (x>=numElem.width )
        return;

    float a = 0;
    for (unsigned fi=0; fi<numElem.height; fi++)
    {
        a += in_wavelet[ x + fi*numElem.width ].x;
    }

    out_inverse_waveform[x] = a;
}

void wtClamp( float2* in_wt, cudaExtent in_numElem, size_t in_offset, size_t last_sample, float2* out_clamped_wt, cudaExtent out_numElem, cudaStream_t stream )
{
    // in this scope, work on arrays of float* instead of float2* to coalesce better
    in_numElem.width *= 2;
    in_offset *= 2;
    out_numElem.width *= 2;
    last_sample *= 2;

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

    kernel_clamp<<<grid, block, stream>>>( (float*)in_wt, in_numElem, in_offset, last_sample, (float*)out_clamped_wt, out_numElem );
}

__global__ void kernel_clamp( float* in_wt, cudaExtent in_numElem, size_t in_offset, size_t last_sample, float* out_clamped_wt, cudaExtent out_numElem )
{
    const unsigned
            x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x,
            y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    // sanity checks...
    if (x>=out_numElem.width )
        return;
    if (y>=out_numElem.height)
        return;

    // Not coalesced reads for arbitrary in_offset, coalesced writes though
    float v = 0;
    if (y<in_numElem.height && in_offset + x < in_numElem.width) {
        unsigned i = in_offset + x + in_numElem.width*y;
        v = in_wt[i];
    }

    if (x >= last_sample)
        v = 0.f/0.f;

    unsigned o = x + out_numElem.width*y;
    out_clamped_wt[o] = v;
}
