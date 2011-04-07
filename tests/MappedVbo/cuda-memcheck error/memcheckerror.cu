#include <GL/glew.h>
#include <GL/glut.h>
#include <iostream>
#include <cuda_gl_interop.h>

using namespace std;

#define BLOCK_SIZE 128
__global__ void simpleKernel(
        float* output )
{
    output[threadIdx.x] = 0;
}

cudaGraphicsResource* positionsVBO_CUDA;

void display()
{
    cudaError cuda_inited = cudaGLSetGLDevice(0);
    int glew_inited = glewInit();

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
    cout << "glew_inited = " << (glew_inited == 0) << endl;
    cout << "is_registered = "<< (is_registered == cudaSuccess) << endl;
    cout << "is_mapped = "<< (is_mapped == cudaSuccess) << endl;
    cout << "num_bytes = " << num_bytes << endl;
    cout << "g_data = " << g_data << endl;
    cout << "got_pointer = "<< (got_pointer == cudaSuccess) << endl;
    cout << "unmapped = "<< (unmapped == cudaSuccess) << endl;
    cout << "unreg = "<< (unreg == cudaSuccess) << endl;

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
