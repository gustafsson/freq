#include <GL/glut.h>
#include <iostream>
#include <fstream>

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
	
    unsigned N = BLOCK_SIZE;
    unsigned size = N*sizeof(float);
    float* g_data;
    cudaError mallocd = cudaMalloc( &g_data, size );
    dim3 block( BLOCK_SIZE );
    dim3 grid( 1 );
    simpleKernel<<< grid, block>>>(g_data);
	
    cudaError freed = cudaFree( g_data );
	cudaError sync = cudaThreadSynchronize();

    cout << "mallocd = " << (mallocd == cudaSuccess) << endl
         << "freed = " << (freed == cudaSuccess) << endl
         << "sync = " << (sync == 0) << endl;
	
    string name = __FILE__ " log.txt";

    ofstream tst(name.c_str());
    tst << name.c_str() << endl
         << "mallocd = " << (mallocd == cudaSuccess) << endl
         << "freed = " << (freed == cudaSuccess) << endl
         << "sync = " << (sync == 0) << endl;
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
