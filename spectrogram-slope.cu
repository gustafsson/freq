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

    if (x>=width || y>=height)
        return;

    unsigned int i = y*width+x;

    int top=-1, left=-1, bottom=1, right=1;

    // clamp
    if (x == 0)
        left = 0;
    if (y == 0)
        top = 0;
    if (x == width-1)
        right = 0;
    if (y == height-1)
        bottom = 0;

    float2 slope = make_float2(
        (h[i + right] - h[i + left])/(right-left),
        (h[i + width*bottom] - h[i + width*top])/(bottom-top));
    slopeOut[i] = slope;
}


extern "C"
void cudaCalculateSlopeKernel(  float* hptr, float2 *slopeOut,
                                unsigned int width, unsigned int height, unsigned cuda_stream)
{
    dim3 block(8, 8, 1);
    dim3 grid2(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
    calculateSlopeKernel<<<grid2, block, cuda_stream>>>(hptr, slopeOut, width, height);
}
