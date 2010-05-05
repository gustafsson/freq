#include "cudaUtil.h"
#include <stdio.h>
#include "wavelet.cu.h"

__global__ void kernel_compute( float* in_waveform_ft, float* out_wavelet_ft, cudaExtent numElem, float start, float scales_per_octave, float steplogsize  );
__global__ void kernel_inverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, unsigned n_valid_samples );
__global__ void kernel_inverse_ellips( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples );
__global__ void kernel_inverse_box( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples );
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

void wtCompute( float2* in_waveform_ft, float2* out_wavelet_ft, unsigned sampleRate, float minHz, float maxHz, cudaExtent numElem, float scales_per_octave, cudaStream_t stream )
{
    // in this scope, work on arrays of float* instead of float2* to coalesce better
    numElem.width *= 2;

    float start = sampleRate/minHz/numElem.width;
    float steplogsize = log(maxHz)-log(minHz);

    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), numElem.depth, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

	// float scales_per_octave = numElem.height/((log(maxHz)/log(2.f)-(log(minHz)/log(2.f));
    kernel_compute<<<grid, block, 0, stream>>>( (float*)in_waveform_ft, (float*)out_wavelet_ft, numElem, start, steplogsize, scales_per_octave );
}

__global__ void kernel_compute(
        float* in_waveform_ft,
        float* out_wavelet_ft,
        cudaExtent numElem, float start, float steplogsize, float scales_per_octave )
{
    // Element number
    const unsigned
            x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (x>=numElem.width)
        return;

    float waveform = in_waveform_ft[x];

    float cufft_normalize = rsqrt((float)numElem.width);
    float jibberish_normalization =  26.0993675428f/scales_per_octave;

    // Find period for this thread
    unsigned nFrequencies = numElem.height;
    unsigned channel = blockIdx.y;
    unsigned n = numElem.width;
    for( unsigned fi = 0; fi<nFrequencies; fi++) {
        float ff = fi/(float)nFrequencies;
        float period = start*exp(-ff*steplogsize);

        // Find offset for this wavelet scale
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

void wtInverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, unsigned n_valid_samples, cudaStream_t stream )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), 1, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_inverse<<<grid, block, 0, stream>>>( in_wavelet, out_inverse_waveform, numElem, n_valid_samples );
}

__global__ void kernel_inverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, unsigned n_valid_samples )
{
    const unsigned
            //x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
            x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x>=n_valid_samples)
        return;
    if (x>=numElem.width )
        return;

    float a = 0;

    // no selection
    for (unsigned fi=0; fi<numElem.height; fi++)
    {
        a += in_wavelet[ x + fi*numElem.width ].x;
    }

    float cufft_normalize = 1.f/sqrt((float)numElem.width);

    // TODO wtf?
    float jibberish_normalization = .1;

    out_inverse_waveform[x] = jibberish_normalization*cufft_normalize*a;
}

void wtInverseEllips( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples, cudaStream_t stream )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), 1, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_inverse_ellips<<<grid, block, 0, stream>>>( in_wavelet, out_inverse_waveform, numElem, area, n_valid_samples );
}

__global__ void kernel_inverse_ellips( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples )
{
    const unsigned
            //x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
            x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x>=n_valid_samples)
        return;
    if (x>=numElem.width )
        return;

    float a = 0;

    // disc selection
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

void wtInverseBox( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples, cudaStream_t stream )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), 1, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_inverse_box<<<grid, block, 0, stream>>>( in_wavelet, out_inverse_waveform, numElem, area, n_valid_samples );
}

__global__ void kernel_inverse_box( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples )
{
    const unsigned
            //x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
            x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x>=n_valid_samples)
        return;
    if (x>=numElem.width )
        return;

    float a = 0;

    // box selection
    if (x>=area.x && x<=area.z)
      {
        for (unsigned fi=max(0.f,area.y); fi<numElem.height && fi<area.w; fi++)
        {
            // select only the real component of the complex transform
            a += in_wavelet[ x + fi*numElem.width ].x;
        }
    }

    float cufft_normalize = 1.f/sqrt((float)numElem.width);
    float jibberish_normalization = .1;

    out_inverse_waveform[x] = jibberish_normalization*cufft_normalize*a;
}
