#include "cudaUtil.h"
#include <stdio.h>

__global__ void kernel_compute( float* in_waveform_ft, float* out_wavelet_ft, cudaExtent numElem, float start, float steplogsize  );
__global__ void kernel_inverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area );
__global__ void kernel_clamp( float* in_wt, cudaExtent in_numElem, size_t in_offset, size_t last_sample, float* out_clamped_wt, cudaExtent out_numElem );
__global__ void kernel_remove_disc(float2* in_wavelet, cudaExtent in_numElem, float4 area );

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
    dim3 grid( INTDIV_CEIL(numElem.width, block.x), numElem.depth, 1);

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
    // Element number
    const unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (x>=numElem.width)
        return;

    float waveform = in_waveform_ft[x];

    float cufft_normalize = 1.f/sqrt((float)numElem.width);
    float jibberish_normalization = 0.05123456;

    // Find period for this thread
    unsigned nFrequencies = numElem.height;
    for( unsigned fi = 0; fi<nFrequencies; fi++) {
        float ff = fi/(float)nFrequencies;
        float period = start*exp(-ff*steplogsize);

        // Find offset for this wavelet scale
        unsigned channel = blockIdx.y;
        unsigned n = numElem.width;
        unsigned offset = fi*n + channel*n*nFrequencies;


        // Compute value of analytic FT of wavelet
        const float f0 = .6f + 40*ff*ff*ff;
        const float pi = 3.141592654f;
        const float two_pi_f0 = 2.0f * pi * f0;
        const float multiplier = 1.8827925275534296252520792527491f;

        period *= f0;

        unsigned y = x/2; // compute equal results for the complex and scalar part
        float factor = 4*pi*y*period-two_pi_f0;
        float basic = multiplier * exp(-0.5f*factor*factor);

        float m = jibberish_normalization*cufft_normalize*basic*f0;
        //float m = cufft_normalize*basic*f0;
        //float m = basic*f0;
        out_wavelet_ft[offset + x] = m * waveform;
    }
}

void wtInverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, cudaStream_t stream )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(numElem.width, block.x), 1, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_inverse<<<grid, block, stream>>>( in_wavelet, out_inverse_waveform, numElem, area );
}

__global__ void kernel_inverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area )
{
    const unsigned
            //x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
            x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x>=numElem.width )
        return;

    float a = 0;
/* no selection
    for (unsigned fi=0; fi<numElem.height; fi++)
    {
        a += in_wavelet[ x + fi*numElem.width ].x;
    }*/
 /* box selection
    if (x>=area.x && x<=area.z)
      {
        for (unsigned fi=max(0.f,area.y); fi<numElem.height && fi<area.w; fi++)
        {
            // select only the real component of the complex transform
            a += in_wavelet[ x + fi*numElem.width ].x;
        }
    }*/
/* disc selection */
    for (unsigned fi=0; fi<numElem.height; fi++)
    {
        float rx = area.z-area.x;
        float ry = area.w-area.y;
        float dx = x-area.x;
        float dy = fi-area.y;

        if (dx*dx/rx/rx + dy*dy/ry/ry < 1) {
            // select only the real component of the complex transform
            a += in_wavelet[ x + fi*numElem.width ].x;
        }
    }

    float cufft_normalize = 1.f/sqrt((float)numElem.width);
    float jibberish_normalization = .1;

    out_inverse_waveform[x] = jibberish_normalization*cufft_normalize*a;
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

    // TODO Not coalesced reads for arbitrary in_offset, coalesced writes though
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

void removeDisc( float2* wavelet, cudaExtent numElem, float4 area )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(numElem.width, block.x), numElem.height*numElem.depth, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_remove_disc<<<grid, block>>>( wavelet, numElem, area );
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
    float dx = x-(float)area.x;
    float dy = fi-(float)area.y;

    if (dx*dx/rx/rx + dy*dy/ry/ry < 1) {
        wavelet[ x + fi*numElem.width ].x = 0;
        wavelet[ x + fi*numElem.width ].y = 0;
    }
}
