// "This software contains source code provided by NVIDIA Corporation."
// from oceanFFT example

//Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b)
{
    return (a + (b - 1)) / b;
}


// generate slope by partial differences in spatial domain
__global__ void calculateSlopeKernel(float* h, float2 *slopeOut, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int i = y*width+x;

    float2 slope;
    if ((x > 0) && (y > 0) && (x < width-1) && (y < height-1)) {
        slope.x = h[i+1] - h[i-1];
        slope.y = h[i+width] - h[i-width];
    } else {
        slope = make_float2(0.0f, 0.0f);
    }
    slopeOut[i] = slope;
}


extern "C"
void cudaCalculateSlopeKernel(  float* hptr, float2 *slopeOut,
                                unsigned int width, unsigned int height)
{
    dim3 block(8, 8, 1);
    dim3 grid2(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    calculateSlopeKernel<<<grid2, block>>>(hptr, slopeOut, width, height);
}
