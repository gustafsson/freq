#include <iostream>
#include <fstream>

using namespace std;

#define BLOCK_SIZE 128
__global__ void simpleKernel(
        float* output )
{
    output[threadIdx.x] = 0;
}


int main(int argc, char *argv[])
{
    unsigned N = BLOCK_SIZE;
    unsigned size = N*sizeof(float);
    float* g_data;
    cudaError mallocd = cudaMalloc( &g_data, size );
    dim3 block( BLOCK_SIZE );
    dim3 grid( 1 );
    simpleKernel<<< grid, block>>>(g_data);
    cudaError freed = cudaFree( g_data );

	cudaError sync = cudaThreadSynchronize();

    bool all_success = (mallocd == cudaSuccess)
		&& (freed == cudaSuccess)
        && (sync == cudaSuccess);

    cout << "mallocd = " << (mallocd == cudaSuccess) << endl
         << "freed = " << (freed == cudaSuccess) << endl
         << "sync = " << (sync == cudaSuccess) << endl;
	
    bool any_failed = !all_success;
    exit(any_failed);
}
