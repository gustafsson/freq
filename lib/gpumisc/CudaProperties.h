#include <cuda_runtime.h>

/**
Wrapper for functions in the cuda runtime API that are used to report
properties of cuda devices.
*/
class CudaProperties {
private:
	~CudaProperties();
public:
	/**
	Wrapper for cudaGetDeviceProperties(...). Documentation copied 
	from CudaReferenceManual.pdf version 2.0.
	<p>
	Returns the properties of device dev (current device if -1==dev).
	The cudaDeviceProp structure is defined as:
	<code>
	struct cudaDeviceProp {
		char name[256];
		size_t totalGlobalMem;
		size_t sharedMemPerBlock;
		int regsPerBlock;
		int warpSize;
		size_t memPitch;
		int maxThreadsPerBlock;
		int maxThreadsDim[3];
		int maxGridSize[3];
		size_t totalConstMem;
		int major;
		int minor;
		int clockRate;
		size_t textureAlignment;
		int deviceOverlap;
		int multiProcessorCount;
	}
	</code>
	where:
	<p><b>name</b>
	is an ASCII string identifying the device.

	<p><b>totalGlobalMem</b>
	is the total amount of global memory available on the device in 
	bytes.

	<p><b>sharedMemPerBlock</b>
	is the maximum amount of shared memory available to a thread 
	block in bytes. This amount is shared by all thread blocks 
	simultaneously resident on a multiprocessor.

	<p><b>regsPerBlock</b>
	is the maximum number of 32-bit registers available to a thread 
	block. This number is shared by all thread blocks simultaneously 
	resident on a multiprocessor.

	<p><b>warpSize</b>
	is the warp size in threads.

	<p><b>memPitch</b>
	is the maximum pitch in bytes allowed by the memory copy 
	functions that involve memory regions allocated through 
	cudaMallocPitch().

	<p><b>maxThreadsPerBlock</b>
	is the maximum number of threads per block.

	<p><b>maxThreadsDim[3]</b>
	is the maximum sizes of each dimension of a block.

	<p><b>maxGridSize[3]</b>
	is the maximum sizes of each dimension of a grid.

	<p><b>totalConstMem</b>
	is the total amount of constant memory available on the device in
	bytes.

	<p><b>major, minor</b>
	are the major and minor revision numbers defining the device’s 
	compute capability.

	<p><b>clockRate/<b>
	is the clock frequency in kilohertz.

	<p><b>textureAlignment</b>
	is the alignment requirement. Texture base addresses that are 
	aligned to textureAlignment bytes do not need an offset applied 
	to texture fetches.

	<p><b>deviceOverlap</b>
	is 1 if the device can concurrently copy memory between host 
	and device while executing a kernel, or 0 if not.

	<p><b>multiProcessorCount</b>
	is the number of multiprocessors on the device.
	*/
	static cudaDeviceProp getCudaDeviceProp( int dev = -1 );

	static void printInfo( cudaDeviceProp );
	static int getCudaCurrentDevice();
    static unsigned getCudaCurrentDeviceMaxFlops();
    static int getCudaDeviceCount();
    static bool haveCuda();

    static double flops(const cudaDeviceProp& prop);
    static double gpu_memory_speed(unsigned *n=0);

    static int getCudaDriverVersion();
    static int getCudaRuntimeVersion();
};
