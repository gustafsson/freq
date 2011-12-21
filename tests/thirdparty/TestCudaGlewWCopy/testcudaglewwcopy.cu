#include <GL/glew.h>
#include <GL/glut.h>
#include <iostream>
#include <fstream>
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
    glClear(GL_COLOR_BUFFER_BIT);
	glutSwapBuffers();
	
    static bool once = true;
	if (!once)
	    return;
    once = false;
	
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

    float* g_data2;
    cudaError mallocd = cudaMalloc( &g_data2, size );
    cudaError memcopied = cudaMemcpy( g_data2, g_data, size, cudaMemcpyDeviceToDevice );

    dim3 block( BLOCK_SIZE );
    dim3 grid( 1 );
    simpleKernel<<< grid, block>>>(g_data2);
    cudaError freed = cudaFree( g_data2 );

    cudaError unmapped = cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
    cudaError unreg = cudaGraphicsUnregisterResource( positionsVBO_CUDA );
	cudaError sync = cudaThreadSynchronize();

    cout << "cuda_inited = " << (cuda_inited == cudaSuccess) << endl
        << "glew_inited = " << (glew_inited == 0) << endl
        << "is_registered = "<< (is_registered == cudaSuccess) << endl
        << "is_mapped = "<< (is_mapped == cudaSuccess) << endl
        << "num_bytes = " << num_bytes << endl
        << "g_data = " << g_data << endl
        << "got_pointer = " << (got_pointer == cudaSuccess) << endl
		<< "mallocd = " << (mallocd == cudaSuccess) << endl
		<< "memcopied = " << (memcopied == cudaSuccess) << endl
		<< "freed = " << (freed == cudaSuccess) << endl
        << "unmapped = "<< (unmapped == cudaSuccess) << endl
        << "unreg = "<< (unreg == cudaSuccess) << endl
        << "sync = "<< (sync == cudaSuccess) << endl;
	
    string name = __FILE__ " log.txt";

    ofstream tst(name.c_str());
    tst << name.c_str() << endl
        << "cuda_inited = " << (cuda_inited == cudaSuccess) << endl
        << "glew_inited = " << (glew_inited == 0) << endl
        << "is_registered = "<< (is_registered == cudaSuccess) << endl
        << "is_mapped = "<< (is_mapped == cudaSuccess) << endl
        << "num_bytes = " << num_bytes << endl
        << "g_data = " << g_data << endl
        << "got_pointer = "<< (got_pointer == cudaSuccess) << endl
		<< "mallocd = " << (mallocd == cudaSuccess) << endl
		<< "memcopied = " << (memcopied == cudaSuccess) << endl
		<< "freed = " << (freed == cudaSuccess) << endl
        << "unmapped = "<< (unmapped == cudaSuccess) << endl
        << "unreg = "<< (unreg == cudaSuccess) << endl
        << "sync = "<< (sync == cudaSuccess) << endl;
}


int main(int argc, char *argv[])
{
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(300, 200);
    glutCreateWindow(__FILE__);
    glutDisplayFunc( display );
    glutMainLoop();
    return 0;
}
