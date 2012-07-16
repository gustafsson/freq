#include "neat_math.h"

__global__ void kernel_memset_fix(
        float2* p,
        int N)
{
    const int
            i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i>=N)
        return;

    p[i] = make_float2(0,0);
}


void cudaMemsetFix(void* p, int N)
{
    if (N%sizeof(float2))
    {
        cudaMemset(p, 0, N);
        return;
    }

    N /= sizeof(float2);

    dim3 block(64,1,1);
    dim3 grid( int_div_ceil(N, block.x), 1, 1);

    int L = 32768;
    if(grid.x>L) {
        int skipBytes = block.x*L*sizeof(float2);
        cudaMemsetFix( ((char*)p) + skipBytes, N*sizeof(float2) - skipBytes);
        grid.x = L;
    }

    kernel_memset_fix<<<grid, block>>>((float2*)p, N);
}
