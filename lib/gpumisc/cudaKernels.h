/**
cudaKernels.h contains a bunch of useful inline functions to use in
cuda kernels. It has been undergone a refactoring so some parts
should be moved out.

@author Johan.b.Gustafsson@gmail.com
@date 2008-11-17
*/

/*texture<Color_t, 1, cudaReadModeElementType> tex_colorMap;
*/
#define VISIBLE_GRADIENT 1
#define VISIBLE_GRADIENT_INV (1.f/1.f)
#define POSITION_RANGE 8192
#define POSITION_RANGEf 8192.f
#define POSITION_RANGE_INV (1.f/POSITION_RANGEf)
#define INV4095 (1.f/4095.f)

// must be a power of 2 for the FindMinMax_kernel algorithm to work
#define FIND_MINMAX_N_THREADS 128 

#define VALUES_PER_THREAD (sizeof(uint4)/sizeof(VolumeSource_t))

#include <misc/stringprintf.h>

typedef unsigned short ushort;
typedef unsigned int uint;

#  define DEVICE_INIT() do {                                                  \
	int argc = 1;                                                             \
	const char* argv[]={""};                                                  \
	CUT_DEVICE_INIT(argc,argv);                                               \
	} while(0)

#  define LOG_FAILURE(who, what, why) do {                                    \
    cerrprintf( why );                          \
   } while(0)

#  define LOG_FAILURE2(who, what, why) do {                                    \
    cerrprintf("%s error in file '%s' at line %i "                            \
               "when executing \"%s\": %s\n"                                  \
               who, __FILE__, __LINE__, what, why );                          \
   } while(0)

#  define CUDA_RETURN_ON_FAIL( call ) do {                                    \
    cudaError err = (call);                                                   \
    if (cudaSuccess != err) {                                                 \
	    LOG_FAILURE("Cuda", "", cudaGetErrorString( err) );                \
        return;                                                               \
    } } while(0)

#  define CUDA_RETURN_ON_VOID_FAIL_VALUE( voidCall, failValue ) do {          \
    voidCall;                                                                 \
    cudaError err = cudaGetLastError();                                       \
    if (cudaSuccess != err) {                                                 \
		LOG_FAILURE("Cuda", "", cudaGetErrorString( err) );                   \
        return failValue;                                                     \
    } } while(0)

#  define CUDA_RETURN_ON_VOID_FAIL( voidCall )                                \
	CUDA_RETURN_ON_VOID_FAIL_VALUE( voidCall, ; );

#  define CUT_RETURN_ON_FAIL( call) do {                                      \
    if( CUTTrue != call) {                                                    \
	    LOG_FAILURE("Cut", #call, " != CUTTrue" );                            \
        return;                                                               \
	} } while(0)

// Note: watch out for integer overflow when creating ushort2
__device__ inline ushort2 threadPosX_Y( )
{
	const ushort
		tx = blockIdx.x*blockDim.x + threadIdx.x,
		ty = blockIdx.y*blockDim.y + threadIdx.y;
	return make_ushort2(		tx,		ty );
}

// Note: watch out for integer overflow when creating ushort3
__device__ inline ushort3 threadPosX_Y_Z( )
{
	const unsigned 
		tx = blockIdx.x*blockDim.x + threadIdx.x,
		ty = blockIdx.y*blockDim.y + threadIdx.y,
		tz = blockIdx.z*blockDim.z + threadIdx.z;
	return make_ushort3(		tx,		ty,		tz );
}

__device__ inline bool threadPosValid( const ushort3& pos, const ushort3& totalDim )
{
	return
		pos.x < totalDim.x &&
		pos.y < totalDim.y &&
		pos.z < totalDim.z;
}

// Note: watch out for integer overflow when creating ushort3
__device__ inline ushort3 threadPosXY_Z( const ushort& xsize )
{
	const unsigned 
		tx = blockIdx.x*blockDim.x + threadIdx.x,
		ty = blockIdx.y*blockDim.y + threadIdx.y;
	return make_ushort3(		tx%xsize,		tx/xsize,	ty );
}

// Note: watch out for integer overflow when creating ushort3
__device__ inline ushort3 threadPosX_YZ( const ushort& ysize )
{
	const unsigned 
		tx = blockIdx.x*blockDim.x + threadIdx.x,
		ty = blockIdx.y*blockDim.y + threadIdx.y;
	return make_ushort3(		tx,		ty%ysize,		ty/ysize );
}

__device__ inline unsigned int threadId()
{
	return blockIdx.x*blockDim.x + threadIdx.x
		+ gridDim.x*blockDim.x* ((blockIdx.y*blockDim.y + threadIdx.y)
												   + gridDim.y*blockDim.y* (blockIdx.z*blockDim.z + threadIdx.z));
}

__device__ inline unsigned int blockId()
{
	return blockIdx.x +
		gridDim.x * (blockIdx.y +
						    gridDim.y * ( blockIdx.z ));
}

template<typename t>
__device__ inline t clamp( const t& a, const t& v, const t& b )
{
	return max(a,min(v,b));
}

__device__ inline unsigned int offs(  const ushort3& sz, const short3& pos )
{
	return clamp((short)0,pos.x,(short)(sz.x-1)) 
		+ sz.x * (clamp((short)0,pos.y,(short)(sz.y-1)) 
		+ sz.y * clamp((short)0,pos.z,(short)(sz.z-1)));
}
__device__ inline unsigned int offs(  const ushort3& sz, const ushort3& pos )
{
	return min(pos.x,(ushort)(sz.x-1)) 
		+ sz.x * (min(pos.y,(ushort)(sz.y-1)) 
		+ sz.y * min(pos.z,(ushort)(sz.z-1)));
}
// NOTE! size is the first argument and pos is the second
__device__ inline unsigned int o(  const ushort3& sz, const ushort3& pos )
{
	return pos.x
		+ sz.x * (pos.y 
		+ sz.y * pos.z);
}
