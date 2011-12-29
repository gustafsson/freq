#include "gl.h"
#ifndef __APPLE__
#   include <GL/glut.h>
#else
#   include <GLUT/glut.h>
#endif
#include "cudaPitchedPtrType.h"
#include <iostream>
#include <cuda_gl_interop.h>
#include "vbo.h"

using namespace std;

#define BLOCK_SIZE 128
__global__ void mappedVboTestKernel(
        float* output, unsigned sz)
{
    unsigned writePos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (writePos<sz)
    {
        unsigned o = writePos;
        output[o] = 0;
    }
}

#define BLOCK_SIZE 128
__global__ void simpleKernel(
        float* output )
{
    //output[threadIdx.x] = 0;
    output[0] = 0;
}

cudaGraphicsResource* positionsVBO_CUDA;

void display()
{
    cudaError cuda_inited = cudaGLSetGLDevice(0);
#ifndef __APPLE__ // glewInit is not needed on Mac
    int glew_inited = glewInit();
#endif

    unsigned N = BLOCK_SIZE;
    unsigned size = N*sizeof(float);
    unsigned vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaError is_registered = cudaGraphicsGLRegisterBuffer( &positionsVBO_CUDA, vbo, cudaGraphicsMapFlagsWriteDiscard);
    cudaError is_mapped = cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0 );
    float* g_data;
    size_t num_bytes;
    cudaError got_pointer = cudaGraphicsResourceGetMappedPointer((void**)&g_data, &num_bytes, positionsVBO_CUDA);

    dim3 block( BLOCK_SIZE );
    dim3 grid( 1 );
    simpleKernel<<< grid, block>>>(g_data);

    cudaError unmapped = cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
    cudaError unreg = cudaGraphicsUnregisterResource( positionsVBO_CUDA );

    cout << "cuda_inited = " << (cuda_inited == cudaSuccess) << endl;
#ifndef __APPLE__ // glewInit is not needed on Mac
    cout << "glew_inited = " << (glew_inited == 0) << endl;
#endif
    cout << "is_registered = "<< (is_registered == cudaSuccess) << endl;
    cout << "is_mapped = "<< (is_mapped == cudaSuccess) << endl;
    cout << "num_bytes = " << num_bytes << endl;
    cout << "g_data = " << g_data << endl;
    cout << "got_pointer = "<< (got_pointer == cudaSuccess) << endl;
    cout << "unmapped = "<< (unmapped == cudaSuccess) << endl;
    cout << "unreg = "<< (unreg == cudaSuccess) << endl;

    ::exit(0);
}

void displayOld()
{
    cout << "disp" << endl;
    cudaError cuda_inited = cudaGLSetGLDevice(0);
    cout << "cuda_inited = " << (cuda_inited == cudaSuccess) << endl;
#ifndef __APPLE__ // glewInit is not needed on Mac
    int glew_inited = glewInit();
    cout << "glew_inited = " << (glew_inited == 0) << endl;
#endif

    unsigned N = 256;
    unsigned size = N*sizeof(float);
    unsigned vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGLRegisterBufferObject(vbo);
    float* g_data=0;
    cudaError is_mapped = cudaGLMapBufferObject((void**)&g_data, vbo);
    cout << "is_mapped = "<< (is_mapped == cudaSuccess) << endl;

    dim3 block( BLOCK_SIZE );
    dim3 grid( (N + block.x-1)/block.x );
    mappedVboTestKernel<<< grid, block>>>(g_data, N);

    cudaError unmapped = cudaGLUnmapBufferObject(vbo);
    cudaError unreg = cudaGLUnregisterBufferObject(vbo);
    cout << "unmapped = "<< (unmapped == cudaSuccess) << endl;
    cout << "unreg = "<< (unreg == cudaSuccess) << endl;

    /*float*devptr;
    cudaMalloc(&devptr, sizeof(float)*32);
    mappedVboTestKernel<<< grid, block>>>(devptr, sz_output);
    cudaFree(devptr);*/
    ::exit(0);
}

int main(int argc, char *argv[])
{
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(300, 200);
    glutCreateWindow("Mapped VBO test");
    glutDisplayFunc( display );
    glutMainLoop();
    return 0;
}

#if 0
#include "cudaPitchedPtrType.h"

__global__ void mappedVboTestKernel(
        float* output, elemSize3_t sz)
{
    elemSize3_t  writePos;
    writePos.x = blockIdx.x * 128 + threadIdx.x;
    writePos.y = blockIdx.y * 1 + threadIdx.y;
    if (writePos.x<sz.x && writePos.y < sz.y)
    {
        unsigned o = writePos.x  +  writePos.y * sz.x;
        o = o % 32;
        output[o] = 0;
    }
}


void mappedVboTestCuda( cudaPitchedPtrType<float> data )
{
    elemSize3_t sz_output = data.getNumberOfElements();
    dim3 block( 128 );
    dim3 grid( int_div_ceil( sz_output.x, block.x ), sz_output.y );
    mappedVboTestKernel<<< grid, block>>>(data.ptr(), sz_output);
    float*devptr;
    cudaMalloc(&devptr, sizeof(float)*32);
    //mappedVboTestKernel<<< grid, block>>>(devptr, sz_output);
    cudaFree(devptr);
    return;
}
#endif
