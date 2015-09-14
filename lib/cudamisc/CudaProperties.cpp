#include "CudaProperties.h"
#include "CudaException.h"
#include "tasktimer.h"
#include "cpuproperties.h"

#include <cuda_runtime.h>

#include <boost/date_time/posix_time/posix_time.hpp>


cudaDeviceProp CudaProperties::
        getCudaDeviceProp( int device )
{
    if (-1 == device ) {
        device = getCudaCurrentDevice();
    }
    cudaDeviceProp prop;
    CudaException_SAFE_CALL( cudaGetDeviceProperties ( &prop, device ) );
    return prop;
}


void CudaProperties::
        printInfo( cudaDeviceProp p )
{
    TaskTimer tt("Cuda device properties for %s", p.name);

    tt.info("sharedMemPerBlock = %lu", p.sharedMemPerBlock );
    tt.info("totalGlobalMem = %lu", p.totalGlobalMem);
    tt.info("regsPerBlock = %d", p.regsPerBlock );
    tt.info("warpSize = %d", p.warpSize );
    tt.info("memPitch = %lu", p.memPitch );
    tt.info("maxThreadsPerBlock = %d, maxThreadsDim = {%d, %d, %d}", p.maxThreadsPerBlock, p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    tt.info("maxGridSize = {%d, %d, %d}", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    tt.info("clockRate = %d kHz = %.1f GHz", p.clockRate, p.clockRate*1e-6);
    tt.info("totalConstMem = %lu", p.totalConstMem);
    tt.info("major = %d, minor = %d", p.major, p.minor );
    tt.info("textureAlignment = %lu", p.textureAlignment );
    tt.info("deviceOverlap = %d", p.deviceOverlap);
    tt.info("multiProcessorCount = %d", p.multiProcessorCount );

#if 3000 <= CUDART_VERSION
    tt.info("canMapHostMemory = %d", p.canMapHostMemory);
    switch(p.computeMode) {
    case cudaComputeModeDefault: tt.info("computeMode = default"); break;
    case cudaComputeModeExclusive: tt.info("computeMode = exclusive"); break;
    case cudaComputeModeProhibited: tt.info("computeMode = prohibited"); break;
    default: tt.info("computeMode = {invalid}"); break;
    }
    tt.info("concurrentKernels = %d", p.concurrentKernels);
    tt.info("integrated = %d", p.integrated);
    tt.info("kernelExecTimeoutEnabled = %d", p.kernelExecTimeoutEnabled);
    tt.info("maxTexture1D = %lu", (1<<27) );
    tt.info("maxTexture1DArray = %d", p.maxTexture1D );
    tt.info("maxTexture2D = {%d, %d}", p.maxTexture2D[0], p.maxTexture2D[1] );
#if 4000 > CUDART_VERSION
    tt.info("maxTexture2DArray = {%d, %d, %d}", p.maxTexture2DArray[0], p.maxTexture2DArray[1], p.maxTexture2DArray[2] );
    tt.info("maxTexture3D = {%d, %d, %d}", p.maxTexture3D[0], p.maxTexture3D[1], p.maxTexture3D[2] );
#endif
#endif
#if 3000 < CUDART_VERSION
    tt.info("ECCEnabled = %d", p.ECCEnabled);
    tt.info("pciBusID = %d", p.pciBusID );
    tt.info("pciDeviceID = %d", p.pciDeviceID );
    tt.info("surfaceAlignment = %lu", p.surfaceAlignment );
#endif
#if 3010 < CUDART_VERSION
    tt.info("tccDriver = %d", p.tccDriver );
#endif

	tt.info("");



    tt.info("gigaflops = %g",  flops(p) / 1e9);
	{
        unsigned n;
        double bps = gpu_memory_speed(&n);
        cudaError_t e = cudaGetLastError();

        if (e != cudaSuccess)
            TaskInfo("device memory speed: failed %s", cudaGetErrorString( e ) );
        else
            TaskInfo("device memory speed: %.1f GB/s, %u bytes written, %g seconds", bps*1e-9, n, n/bps);
    }
	{
        unsigned n;
        double bps = CpuProperties::cpu_memory_speed(&n);
        TaskInfo("cpu memory speed: %.1f GB/s, %u bytes written, %g seconds", bps*1e-9, n, n/bps);
	}

	tt.info("getCudaDeviceCount = %u", getCudaDeviceCount());
	tt.info("getCudaDriverVersion = %u", getCudaDriverVersion());
	tt.info("getCudaRuntimeVersion = %u", getCudaRuntimeVersion());
}


double CudaProperties::
        flops(const cudaDeviceProp& p)
{
    // 5.4.1 Arithmetic Instructions - CUDA_C_Programming_Guide.pdf
    unsigned flops_per_cycle = 0;
    if (1==p.major)
        flops_per_cycle = 8;
    else if (2==p.major)
    {
        if (0==p.minor)
            flops_per_cycle = 32;
        if (1==p.minor)
            flops_per_cycle = 48;
    }
    else if (3==p.major)
        flops_per_cycle = 192;

    return p.clockRate*1e3 * p.multiProcessorCount*flops_per_cycle;
}


double CudaProperties::
        gpu_memory_speed(unsigned *sz)
{
    void *a, *b;
    unsigned n = (1<<26);
    unsigned M = 10;
    if (sz)
        *sz = 2*M*n;
    cudaMalloc( &a, n );
    cudaMalloc( &b, n );
    cudaMemset( a, 0x3e, n );
    cudaMemcpy( b, a, n, cudaMemcpyDeviceToDevice ); // cold run
    cudaThreadSynchronize();
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    for (unsigned m=0; m<M; ++m)
        cudaMemcpy( b, a, n, cudaMemcpyDeviceToDevice ); // warm run
    cudaThreadSynchronize();
    boost::posix_time::time_duration d = boost::posix_time::microsec_clock::local_time() - start;
    double dt = d.total_microseconds()*1e-6;

    cudaFree( b );
    cudaFree( a );
    return 2*M*n/dt;
}


int CudaProperties::
        getCudaCurrentDevice()
{
    int dev;
    CudaException_SAFE_CALL( cudaGetDevice( &dev ) );
    return dev;
}


unsigned CudaProperties::
        getCudaCurrentDeviceMaxFlops()
{
    unsigned n=getCudaDeviceCount();
    unsigned maxflops=0, maxflops_device=(unsigned)-1;

    for (unsigned i=0; i<n; i++)
    {
        cudaDeviceProp p = getCudaDeviceProp( i );
        if (p.clockRate * p.multiProcessorCount > (int)maxflops)
        {
            maxflops_device = i;
            maxflops = p.clockRate * p.multiProcessorCount;
        }
    }

    return maxflops_device;
}


int CudaProperties::
        getCudaDeviceCount()
{
    int count=0;
    // dont use CudaException_SAFE_CALL, instead count is set to zero if no device is found
    return cudaSuccess==cudaGetDeviceCount( &count ) ? count : 0;
}


bool CudaProperties::
        haveCuda()
{
    cudaDeviceProp prop;
    int dev, count;
    bool r = 0<(count=getCudaDeviceCount());
    r &= (cudaSuccess == cudaGetDevice( &dev ));
    r &= (cudaSuccess == cudaGetDeviceProperties ( &prop, dev ));
    return r;
}


int CudaProperties::
        getCudaDriverVersion()
{
    int n = 0;
#if 2000 >= CUDART_VERSION
    n = 200;
#else
    if (cudaSuccess != cudaDriverGetVersion(&n))
        n = -1;
#endif
    return n;
}


int CudaProperties::
        getCudaRuntimeVersion()
{
    int n = 0;
#if 2000 >= CUDART_VERSION
    n =                                                                     0;
#else
    if (cudaSuccess != cudaRuntimeGetVersion(&n))
        n = -1;
#endif
    return n;
}
