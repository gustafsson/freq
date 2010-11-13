#include "cudaUtil.h"
#include <stdio.h>
#include "tfr/wavelet.cu.h"

__global__ void kernel_compute_wavelet_coefficients( float2* in_waveform_ft, float2* out_wavelet_ft, unsigned nFrequencyBins, unsigned nScales, unsigned first_j, float v, unsigned half_sizes, float sigma_t0 );
__global__ void kernel_inverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, unsigned n_valid_samples );
__global__ void kernel_inverse_ellipse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples );
__global__ void kernel_inverse_box( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples );
__global__ void kernel_clamp( cudaPitchedPtrType<float2> in_wt, size_t sample_offset, cudaPitchedPtrType<float2> out_clamped_wt );

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

#define TOSTR2(x) #x
#define TOSTR(x) TOSTR2(x)
#define setError(x) setError(TOSTR(__FUNCTION__) ": " x)

void wtCompute(
        float2* in_waveform_ft,
        float2* out_wavelet_ft,
        float fs,
        float /*minHz*/,
        float maxHz,
        cudaExtent numElem,
        unsigned half_sizes,
        float scales_per_octave,
        float sigma_t0,
        cudaStream_t stream )
{
//    nyquist = FS/2
//    a = 2 ^ (1/v)
//    aj = a^j
//    hz = fs/2/aj
//    maxHz = fs/2/(a^j)
//    (a^j) = fs/2/maxHz
//    exp(log(a)*j) = fs/2/maxHz
//    j = log(fs/2/maxHz) / log(a)
//    const float log2_a = log2f(2.f) / v = 1.f/v; // a = 2^(1/v)
    float j = (log2(fs/2) - log2(maxHz)) * scales_per_octave;
    unsigned first_scale = max(0.f, floor(j));

    if (j<0) {
        printf("j = %g, maxHz = %g, fs = %g\n", j, maxHz, fs);
        setError("Invalid argument, maxHz must be less than or equal to fs/2.");
        return;
    }

    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), numElem.depth, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_compute_wavelet_coefficients<<<grid, block, 0, stream>>>(
            in_waveform_ft,
            out_wavelet_ft,
            numElem.width, numElem.height,
            first_scale,
            scales_per_octave,
            half_sizes,
            sigma_t0 );
}


/**
  Well, strictly speaking this doesn't produce the 'true' wavelet coefficients
  but rather the coefficients resulting from inversing the wavelet coefficients,
  still in the fourier domain.

  Each thread computes the scale corresponding to the highest frequency first
  and loops down to the scale corresponding to the lowest frequency.

  TODO see matlab file

  @param in_waveform_ft
  Given input signal in fourier domain.

  @param out_wavelet_ft
  Preallocated output coefficients in fourier domain

  @param numElem
  2D size of out_wavelet_ft. numElem.x is size of in_waveform_ft.
  numElem.y is number of scales.

  @param first_scale
  The first scale to compute, first_scale=0 corresponds to the nyquist
  frequency.

  @param v
  Scales per octave is commonly refered to as 'v' in the wavelet bible.

  @param sigma_t0
  Sigma of the mother gabor wavelet in the time domain. Describes the
  time-frequency resolution ratio.
  */
__global__ void kernel_compute_wavelet_coefficients(
        float2* in_waveform_ft,
        float2* out_wavelet_ft,
        unsigned nFrequencyBins, unsigned nScales, unsigned first_scale, float v, unsigned half_sizes, float sigma_t0 )
{
    // Which frequency bin in the discrete fourier transform this thread
    // should work with
    const unsigned
            w_bin = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (w_bin>=nFrequencyBins)
        return;

    const float pi = 3.141592654f;
    const float
            w = w_bin*2*pi/nFrequencyBins;

    float2 waveform_ft;

    if (w_bin>nFrequencyBins/2)
    {
        waveform_ft = make_float2(0,0); // Negative frequencies are defined as 0
    }
    else
    {
        float cufft_normalize = 1.f/(float)(nFrequencyBins*half_sizes);
        float jibberish_normalization = 0.083602;
        cufft_normalize *= jibberish_normalization;

        waveform_ft = in_waveform_ft[w_bin];
        waveform_ft.x *= cufft_normalize;
        waveform_ft.y *= cufft_normalize;
    }

    // Find period for this thread
    const float log2_a = 1.f / v; // a = 2^(1/v)

    float sigma_t0j = sigma_t0; // TODO vary with 'j'
    float sigma_constant = sqrt( 4*pi*sigma_t0j );

    waveform_ft.x *= sigma_constant;
    waveform_ft.y *= sigma_constant;
    for( unsigned j=0; j<nScales; j++)
    {
        // Compute the child wavelet
        // a = 2^(1/v)
        // aj = a^j
        // aj = pow(a,j) = exp(log(a)*j)
        float2 output = make_float2(0,0);
        if (waveform_ft.x != 0 || waveform_ft.y != 0)
        {
            float aj = exp2f(log2_a * (j + first_scale) );

            {
                // Different scales may have different mother wavelets, kind of
                // That is, different sigma_t0j for different j
                // ff = j / (float)total_nScales
                // float f0 = 2.0f + 35*ff*ff*ff
            }
            float q = (-w*aj + pi)*sigma_t0j;
            float phi_star = expf( -q*q );

            output.x = phi_star * waveform_ft.x;
            output.y = phi_star * waveform_ft.y;
        }

        // Find offset for this wavelet coefficient. Writes the scale
        // corresponding to the lowest frequency on the first row of the
        // output matrix
        unsigned offset = (nScales-1-j)*nFrequencyBins;

        // Write wavelet coefficient in output matrix
        out_wavelet_ft[offset + w_bin] = output;
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

    out_inverse_waveform[x] = a;
}

void wtInverseEllipse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples, cudaStream_t stream )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.width, block.x), 1, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_inverse_ellipse<<<grid, block, 0, stream>>>( in_wavelet, out_inverse_waveform, numElem, area, n_valid_samples );
}

__global__ void kernel_inverse_ellipse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples )
{
    const unsigned
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

    out_inverse_waveform[x] = a;
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

    out_inverse_waveform[x] = a;
}

void wtClamp( cudaPitchedPtrType<float2> in_wt, size_t sample_offset, cudaPitchedPtrType<float2> out_clamped_wt, cudaStream_t stream  )
{
    // Multiply the coefficients together and normalize the result

    dim3 grid, block;
    unsigned block_size = 256;
    out_clamped_wt.wrapCudaGrid2D( block_size, grid, block );

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_clamp<<<grid, block, 0, stream>>>( in_wt, sample_offset, out_clamped_wt );
}

__global__ void kernel_clamp( cudaPitchedPtrType<float2> in_wt, size_t sample_offset, cudaPitchedPtrType<float2> out_clamped_wt )
{
    elemSize3_t writePos;
    if( !out_clamped_wt.unwrapCudaGrid( writePos ))
        return;

    elemSize3_t readPos = writePos;
    readPos.x += sample_offset;

    out_clamped_wt.e( writePos ) = in_wt.elem(readPos);
}

