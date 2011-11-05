#ifndef WAVELETKERNEL_CUDA_CU
#define WAVELETKERNEL_CUDA_CU

#include <stdio.h>

#include "resamplecuda.cu.h"
#include "cuda_vector_types_op.h"
#include "waveletkerneldef.h"

__global__ void kernel_compute_wavelet_coefficients( float2* in_waveform_ft, float2* out_wavelet_ft, unsigned nFrequencyBins, unsigned nScales, float first_j, float v, unsigned half_sizes, float sigma_t0, float normalization_factor );
__global__ void kernel_inverse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem );
//__global__ void kernel_inverse_ellipse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples );
//__global__ void kernel_inverse_box( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples );
__global__ void kernel_clamp( cudaPitchedPtrType<float2> in_wt, size_t sample_offset, cudaPitchedPtrType<float2> out_clamped_wt );
__global__ void kernel_stftNormalizeInverse( cudaPitchedPtrType<float> wave, float v );

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

#ifdef _MSC_VER
    #define setError(x) setError(__FUNCTION__ ": " x)
#endif

#if 0
void wtCompute(
        DataStorage<Tfr::ChunkElement>::Ptr in_waveform_ftp,
        Tfr::ChunkData::Ptr out_wavelet_ftp,
        float fs,
        float /*minHz*/,
        float maxHz,
        unsigned half_sizes,
        float scales_per_octave,
        float sigma_t0,
        float normalization_factor )
{
    CudaGlobalStorage::useCudaPitch( out_wavelet_ftp, false );

    Tfr::ChunkElement* in_waveform_ft = CudaGlobalStorage::ReadOnly<1>( in_waveform_ftp ).device_ptr();
    Tfr::ChunkElement* out_wavelet_ft = CudaGlobalStorage::WriteAll<2>( out_wavelet_ftp ).device_ptr();

    DataStorageSize size = out_wavelet_ft->size();

//    nyquist = FS/2
//    a = 2 ^ (1/v)
//    aj = a^j
//    hz = fs/2/aj
//    maxHz = fs/2/(a^j)
//    (a^j) = fs/2/maxHz
//    exp(log(a)*j) = fs/2/maxHz
//    j = log(fs/2/maxHz) / log(a)
//    const float log2_a = log2f(2.f) / v = 1.f/v; // a = 2^(1/v)
    float j = (log2f(fs/2) - log2f(maxHz)) * scales_per_octave;
    float first_scale = j;

    j = floor(j+0.5f);

    if (j<0) {
        printf("j = %g, maxHz = %g, fs = %g\n", j, maxHz, fs);
        setError("Invalid argument, maxHz must be less than or equal to fs/2.");
        return;
    }

    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(size.width, block.x), 1, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_compute_wavelet_coefficients<<<grid, block, 0>>>(
            (float2*)in_waveform_ft,
            (float2*)out_wavelet_ft,
            size.width, size.height,
            first_scale,
            scales_per_octave,
            half_sizes,
            sigma_t0,
            normalization_factor );
}
#endif

__global__ void kernel_compute_wavelet_coefficients(
        float2* in_waveform_ft,
        float2* out_wavelet_ft,
        unsigned nFrequencyBins, unsigned nScales, float first_scale, float v, unsigned half_sizes, float sigma_t0,
        float normalization_factor )
{
    // Which frequency bin in the discrete fourier transform this thread
    // should work with
    const unsigned
            w_bin = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    compute_wavelet_coefficients_elem(
            w_bin,
            in_waveform_ft,
            out_wavelet_ft,
            nFrequencyBins,
            nScales,
            first_scale,
            v,
            half_sizes,
            sigma_t0,
            normalization_factor);
}

#if 0
void wtInverse( Tfr::ChunkData::Ptr in_waveletp, DataStorage<float>::Ptr out_inverse_waveform, DataStorageSize x )
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(x.width, block.x), 1, 1);

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    // kernel_inverse<<<grid, block, 0, stream>>>( in_wavelet, out_inverse_waveform, numElem );
    kernel_inverse<<<grid, block>>>(
            (float2*)CudaGlobalStorage::ReadOnly<2>(in_waveletp).device_ptr(),
            CudaGlobalStorage::WriteAll<1>(out_inverse_waveform).device_ptr(),
            x );
}
#endif

__global__ void kernel_inverse( float2* in_wavelet, float* out_inverse_waveform, DataStorageSize numElem )
{
    const unsigned
            x = blockIdx.x*blockDim.x + threadIdx.x;

    inverse_elem( x, in_wavelet, out_inverse_waveform, numElem );
}


/*
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
            float2 v = in_wavelet[ x + fi*numElem.width ];
            // select only the real component of the complex transform
            a += v.x;
        }
    }

    out_inverse_waveform[x] = a;
}
*/
void wtClamp( Tfr::ChunkData::Ptr in_wtp, size_t sample_offset, Tfr::ChunkData::Ptr out_clamped_wtp )
{
    cudaPitchedPtrType<float2> in_wt(CudaGlobalStorage::ReadOnly<2>( in_wtp ).getCudaPitchedPtr());
    cudaPitchedPtrType<float2> out_clamped_wt(CudaGlobalStorage::WriteAll<2>( out_clamped_wtp ).getCudaPitchedPtr());
    // Multiply the coefficients together and normalize the result

    dim3 grid, block;
    unsigned block_size = 256;
    out_clamped_wt.wrapCudaGrid2D( block_size, grid, block );

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_clamp<<<grid, block, 0>>>( in_wt, sample_offset, out_clamped_wt );
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

void stftNormalizeInverse(
        DataStorage<float>::Ptr wavep,
        unsigned length )
{
    // Multiply the coefficients together and normalize the result
    cudaPitchedPtrType<float> wave(CudaGlobalStorage::ReadWrite<1>( wavep ).getCudaPitchedPtr());

    dim3 grid, block;
    unsigned block_size = 256;
    wave.wrapCudaGrid2D( block_size, grid, block );

    if(grid.x>65535) {
        setError("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    kernel_stftNormalizeInverse<<<grid, block, 0>>>( wave, 1.f/length );
}


__global__ void kernel_stftNormalizeInverse( cudaPitchedPtrType<float> wave, float v )
{
    elemSize3_t writePos;
    if( !wave.unwrapCudaGrid( writePos ))
        return;

    wave.e( writePos ) *= v;
}

#endif // WAVELETKERNEL_CUDA_CU
