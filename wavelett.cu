#include "../misc/inc/cudaUtil.h"

__global__ void WavelettKernel( float* in_waveform_ft, float* out_waveform_ft, float period, unsigned numElem );
__global__ void InverseKernel( float* in_wavelett_ft, float* out_inverse_waveform, cudaExtent numElem );

void computeWavelettTransform( float* in_waveform_ft, float* out_waveform_ft, float period, unsigned numElem );
void inverseWavelettTransform( float* in_wavelett_ft, float* out_inverse_waveform, cudaExtent numElem );

void computeWavelettTransform( float* in_waveform_ft, float* out_waveform_ft, float period, unsigned numElem)
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(numElem, block.x), 1, 1);
    if(grid.x>65535) {
        grid.y = INTDIV_CEIL(grid.x, 65536 );
        grid.x = 65536;
    }

    WavelettKernel<<<grid, block>>>( in_waveform_ft, out_waveform_ft, period, numElem );
}

__global__ void WavelettKernel(
        float* in_waveform_ft,
        float* out_waveform_ft,
        float period, unsigned numElem )
{
    const unsigned
            tx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x,
            ty = __umul24(blockIdx.y,blockDim.y) + threadIdx.y,
            x = ty*blockDim.x*gridDim.x + tx;

    if (x>=numElem)
        return;

    const float f0 = 15;
    const float pi = 3.141592654;
    const float two_pi_f0 = 2.0 * pi * f0;
    const float multiplier = 1.8827925275534296252520792527491;

    period *= f0;

    float factor = 2*pi*x*period-two_pi_f0;
    float basic = multiplier * exp(-0.5*factor*factor);

    out_waveform_ft[x] = in_waveform_ft[x]*basic;
}

void inverseWavelettTransform( float* in_wavelett_ft, float* out_inverse_waveform, cudaExtent numElem)
{
    // Multiply the coefficients together and normalize the result
    dim3 block(256,1,1);
    dim3 grid( INTDIV_CEIL(numElem.width, block.x), 1, 1);
    if(grid.x>65535) {
        grid.y = INTDIV_CEIL(grid.x, 65536 );
        grid.x = 65536;
    }

    InverseKernel<<<grid, block>>>( in_wavelett_ft, out_inverse_waveform, numElem );
}

__global__ void InverseKernel(float* in_wavelett_ft, float* out_inverse_waveform, cudaExtent numElem )
{
    const unsigned
            tx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x,
            ty = __umul24(blockIdx.y,blockDim.y) + threadIdx.y,
            x = ty*blockDim.x*gridDim.x + tx;

    if (x>=numElem.width )
        return;

    float a = 0;
    for (unsigned fi=0; fi<numElem.height; fi++)
    {
        a += in_wavelett_ft[ x + fi*numElem.width ];
    }

    out_inverse_waveform[x] = a;
}
