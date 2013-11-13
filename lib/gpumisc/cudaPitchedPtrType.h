#pragma once

// gpumisc
#include "cudaUtil.h"
#include "neat_math.h"
#include "exceptionassert.h"

typedef uint3 elemSize3_t;
typedef unsigned elemSize_t;
#define make_elemSize3_t make_uint3

// alternate version of cudaPitchedPtr, sizeof(cudaPitchedPtrType)==12 < sizeof(cudaPitchedPtr)==20
// typedef ushort3 elemSize3_t;
// typedef unsigned short pitch_t;
// #define make_elemSize3_t make_ushort3

/**
	Similiar to cudaPitchedPtr, but smaller and comes with a whole bunch of
	neat methods.
*/
template<typename data_t>
struct cudaPitchedPtrType {
#ifndef _DEVICEEMU
private:
#endif
        data_t *device_ptr;
        elemSize3_t elemSize;
        elemSize_t pitch;
public:
    typedef data_t T;

	/**
	*/
	cudaPitchedPtrType():device_ptr(0) {}

	/**
		cudaPitchedPtrType must be created from some cudaPitchedPtr with values
		from one of the cudaMalloc* functions.
	*/
        cudaPitchedPtrType( const cudaPitchedPtr& a, elemSize_t zsize=1);

	/**
		Performs a typecast. The returned cudaPitchedPtrType will point to data
		of the exact same total byte size, but different pointer types and 
		different elemSize.x.
	*/
	template<typename data_t2>
	cudaPitchedPtrType<data_t2> reinterpretCast();


        inline __device__ __host__       data_t* ptr()       { return device_ptr; }
        inline __device__ __host__ const data_t* ptr() const { return device_ptr; }

        inline __device__ __host__ const elemSize3_t& getNumberOfElements() const { return elemSize; }
        inline                           unsigned getTotalBytes() const { return pitch*elemSize.y*elemSize.z; }

    cudaPitchedPtr __device__ __host__ getCudaPitchedPtr() {
        cudaPitchedPtr cpp = {0, 0, 0, 0};
		cpp.ptr = this->ptr();
		cpp.xsize = elemSize.x*sizeof(data_t);
		cpp.ysize = elemSize.y;
		cpp.pitch = pitch;
		return cpp;
	}

	/**
		Gives access to an element of the volume. elem clamps input while e does not.
	*/
    //template<typename vec_t>
    inline __device__ data_t& e( const elemSize3_t& elemPos ) {
        return ptr()[ eOffs( elemPos ) ];
	}

    //template<typename vec_t>
    inline __device__ const data_t& e( const elemSize3_t& elemPos ) const {
        return ptr()[ eOffs( elemPos ) ];
	}
    //template<typename vec_t>
    inline __device__ data_t& elem( elemSize3_t elemPos )     { clamp(elemPos); return e(elemPos); }
    //template<typename vec_t>
    inline __device__ const data_t& elem( elemSize3_t elemPos ) const {	clamp(elemPos); return e(elemPos); }

	/**
		Reads the value of a texture, asserts two things that are valid with 
		the drivers and Cuda version installed on the current development 
		machine. Chances are low (if not zero) that this will change for 
		another 
		current version of Cuda
		<ul>
		<li> Assuming pitch will always be a multiple of 16 bytes. 
		<li> From above it follows that pitch is a multiple of sizeof(data_t),
		because textures can only be bound to "1-, 2-, and 4-component vector
		types" (Cuda Programming Guide). 16 is always a multiple of 
		sizeof("1-, 2-, and 4-component vector types").
		</ul>
		<i>If double4 is to be supported, this method will assume that pitch is a
		multiple of 32 bytes = sizeof(double4).</i>
		<p>
		This coordinate should be used with tex1Dfetch to fetch data. #elemOffs
		clamps the argument properly while #eOffs does not clamp at all.
	*/
	template<typename vec_t>
	inline __device__ unsigned eOffs( const vec_t& elemPos ) const {
		return elemPos.x
			+ pitch/sizeof(data_t) * (elemPos.y 
									 + elemSize.y * elemPos.z);
	}
	template<typename vec_t>
	inline __device__ unsigned elemOffs( vec_t elemPos ) const {
        clamp(elemPos); return eOffs(elemPos);
	}

	/**
		Returns true if param elemPos lies inside the volume, use #clamp to enforce it.
	*/
    __device__ inline bool valid( const uint2& elemPos ) const {
    return elemPos.x<elemSize.x && elemPos.y<elemSize.y; }
        __device__ inline bool valid( const elemSize3_t& elemPos ) const {
		return elemPos.x<elemSize.x && elemPos.y<elemSize.y && elemPos.z<elemSize.z; }

	/**
		Clamps param elemPos to be an element inside the volume, use #valid to check if
		it would be clamped or not.
		<p>
		@param elemPos [in/out] value to clamp. Note that vec_t must not be a const type
		as the argument is both input and output.
	*/
    template<typename vec_t>
    __device__ inline void clamp( vec_t& elemPos ) const;

	/**
		These block and grid sizes should be used together with unampSliceGrid, se below.
		Each wrap is to be used for one slice at a time only. The volume and blocksize must
		satisfy size1/sqrt(blockSize) <= 65535 and size2/sqrt(blockSize) <= 65535, where
		size1 and size2 are the sizes of the two axes in the slice.

		@param blockSize [in] Should be as big as possible, based on kernel complexity.
		@param sliceDim Will align the blocks in x, y or z-planes when used
				together with unampSliceGrid below. sliceDim must be in {0,1,2}.
		@param block [out] Will satisfy block.x*block.y*block.z<=blockSize but be as big
			    as possible.
		@param grid [out] Such that unampSliceGrid umwraps each block appropriately.
	*/
	void wrapSliceGrid( unsigned blockSize, dim3 &grid, dim3& block, char sliceDim = 2 ) const {
                const elemSize3_t nElems = getNumberOfElements();
                const elemSize_t size1 = ((elemSize_t*)&nElems)[(sliceDim+1)%3]; // Pick something else than sliceDim
                const elemSize_t size2 = ((elemSize_t*)&nElems)[(sliceDim+2)%3]; // Pick something else than sliceDim

		// Make sure blockSize is a multiple of warpSize
		const unsigned warpSize = 32;
		if (0<blockSize/warpSize)
			blockSize = blockSize/warpSize*warpSize;

		// Divide the factors of blockSize evenly on block.x and block.y
		unsigned i=2;
		block.x = block.y = block.z = 1;
		while(blockSize>1) {
			if(0==blockSize%i) block.x*=i, blockSize/=i;
			if(0==blockSize%i) block.y*=i, blockSize/=i;
			else i++;
		}

                grid.x = int_div_ceil(size1, block.x);
                grid.y = int_div_ceil(size2, block.y);
		grid.z = 1;
	}

	/**
		Creates a grid that covers the entire volume for the given block size.
		This grid size should be used together with unwrapCudaGrid.

		@param block Given block size to use.
		@returns A grid to be used with unwrapCudaGrid for given block size.
	*/
	dim3 wrapCudaGrid( const dim3& block ) const {
		return ::wrapCudaGrid(getNumberOfElements(), block);
	}

	/**
		These block and grid sizes should be used together with unwrapCudaGrid, se below.
		Blocks will be scanlines, boxes or volumes.

                @param blockSize [in] Should be of adequate size, big but not too big, based on kernel complexity.
		@param block [out] Will satisfy block.x*block.y*block.z<=blockSize but be as big
			    as possible.
		@param grid [out] Such that unwrapCudaGrid umwraps each block appropriately.
	*/
	void wrapCudaGrid1D( unsigned blockSize, dim3 &grid, dim3& block ) const {
		// Make sure blockSize is a multiple of warpSize
		const unsigned warpSize = 32;
		if (0<blockSize/warpSize)
			blockSize = blockSize/warpSize*warpSize;

		block.x = blockSize;
		block.y = block.z = 1;

		grid = wrapCudaGrid(block);
	}

	/**
		@see wrapCudaGrid1D
	*/
	void wrapCudaGrid2D( unsigned blockSize, dim3 &grid, dim3& block ) const {
		// Make sure blockSize is a multiple of warpSize
		const unsigned warpSize = 32;
		if (0<blockSize/warpSize)
			blockSize = blockSize/warpSize*warpSize;

                // Divide the factors of blockSize evenly on block.x and block.y, but
                // first assure block.x>=16 to optimize coalesced memory access
		unsigned i=2;
		block.x = block.y = block.z = 1;
		while(blockSize>1) {
                        if(0==blockSize%i && block.x>16) block.y*=i, blockSize/=i;
                        if(0==blockSize%i) block.x*=i, blockSize/=i;
			else i++;
		}

		grid = wrapCudaGrid(block);
	}

	/**
		@see wrapCudaGrid1D
	*/
	void wrapCudaGrid3D( unsigned blockSize, dim3 &grid, dim3& block ) const {
		// Make sure blockSize is a multiple of warpSize
		const unsigned warpSize = 32;
		if (0<blockSize/warpSize)
			blockSize = blockSize/warpSize*warpSize;

		// Divide the factors of blockSize evenly on block.x, block.y and block.z
		// while maintaining block.z <= 64 (not that it can happen, but still :)).
		unsigned i=2;
		block.x = block.y = block.z = 1;
		while(blockSize>1) {
			if(0==blockSize%i) block.x*=i, blockSize/=i;
			if(0==blockSize%i && block.z*i<=64) block.z*=i, blockSize/=i;
			if(0==blockSize%i) block.y*=i, blockSize/=i;
			else i++;
		}

		grid = wrapCudaGrid(block);
	}

	/**
		@see wrapSliceGrid
	*/
        __device__ inline bool unwrapSliceGrid( elemSize3_t &threadPos, char sliceDim=2, elemSize_t sliceNum=0 )
	{
        #ifdef __CUDACC__
		const unsigned 
            tx = blockIdx.x*blockDim.x + threadIdx.x,
            ty = blockIdx.y*blockDim.y + threadIdx.y;

                elemSize_t* thread = (elemSize_t*)&threadPos;
		thread[(sliceDim+0)%3] = sliceNum;
		thread[(sliceDim+1)%3] = tx;
		thread[(sliceDim+2)%3] = ty;

		return this->valid(threadPos);
	#else 
		return false;  
	#endif
	}

	/**
		Simplification of unwrapCudaMaxGrid, with a known grid wrapping as defined by wrapCudaGrid.
        Uses integer division and integer modulo, _very_ costly operations.
		@see wrapCudaGrid
	*/
        __device__ inline bool unwrapCudaGrid(elemSize3_t &threadPos )
	{
		// Wanted grid size
		// uint3 pitch = {
                //		int_div_ceil(nElems.x, block.x),
                //		int_div_ceil(nElems.y, block.y),
                //		int_div_ceil(nElems.z, block.z)};
		//
		// From wrapCudaGrid this is wrapped to
		// gridDim = {pitch.x*pitch.y, pitch.z, 1};
		//
		// So that 
		// blockIdx = {actualBlockIdx.x+actualBlockIdx.y*pitch.x, actualBlockIdx.z, 1}
		//
		// Or
		// actualBlockIdx = { blockIdx.x % pitchx, blockIdx.x / pitchx, blockIdx.y };
		//
		// And thus
		// threadPos.x = actualBlockIdx.x*blockDim.x + threadIdx.x,
                // threadPos.y = actualBlockIdx.y*blockDim.y + threadIdx.y,
                // threadPos.z = actualBlockIdx.z*blockDim.z + threadIdx.z,

        #ifdef __CUDACC__

		unsigned 
                        pitchx = int_div_ceil(elemSize.x, blockDim.x);
        threadPos.x = (blockIdx.x % pitchx)* blockDim.x + threadIdx.x;
        threadPos.y = (blockIdx.x / pitchx)* blockDim.y + threadIdx.y;
        threadPos.z = blockIdx.y * blockDim.z + threadIdx.z;

		return this->valid(threadPos);
		
	#else
		return false;
	#endif	
	}

	/**
		Using an unknown grid wrapping as defined by wrapCudaMaxGrid.
		@see wrapCudaMaxGrid
	*/
    __device__ inline bool unwrapCudaMaxGrid(elemSize3_t &threadPos )
    {
        #ifdef __CUDACC__
                //const elemSize3_t nElems = getNumberOfElements();
        // Note that gridDim.x does not have anything to do with xPitch for this wrapping.
        unsigned
                        xPitch = int_div_ceil(elemSize.x, blockDim.x),
                        yPitch = int_div_ceil(elemSize.y, blockDim.y),
                        tid;

        unwrapGlobalThreadNumber3D(tid);

        threadPos.z = tid/__umul24(xPitch,yPitch);
        tid %= __umul24(xPitch,yPitch); //equals: tid -= threadPos.z*xPitch*yPitch;
        threadPos.y = tid/xPitch;
        threadPos.x = tid%xPitch;

        return valid(threadPos);
    #else
        return false;
    #endif
    }

    __device__ inline bool unwrapGlobalThreadNumber3D(unsigned& tid)
    {
    #ifdef __CUDACC__
        // global
        tid =  threadIdx.x + __umul24(blockIdx.x,blockDim.x) +
                           + __umul24(__umul24(gridDim.x,blockDim.x),
                                      threadIdx.y + __umul24(blockIdx.y,blockDim.y)
                                                  + __umul24(__umul24(gridDim.y,blockDim.y),
                                                             threadIdx.z + __umul24(blockIdx.z,blockDim.z)
                                                            )
                                      );

        unsigned max = __umul24(elemSize.x, __umul24(elemSize.y, elemSize.y));
        return tid<max;
    #else
        return false;
    #endif
    }
};

template<typename data_t>
cudaPitchedPtrType<data_t>::cudaPitchedPtrType( const cudaPitchedPtr& a, elemSize_t zsize)
{
    pitch = a.pitch;
    EXCEPTION_ASSERT( pitch == a.pitch );
    elemSize.x = int_div_ceil(a.xsize, sizeof(data_t));
    EXCEPTION_ASSERT( elemSize.x == int_div_ceil(a.xsize, sizeof(data_t)) );
    elemSize.y = a.ysize;
    EXCEPTION_ASSERT( elemSize.y == a.ysize );
    elemSize.z = zsize==0?1:zsize;
    EXCEPTION_ASSERT( elemSize.z == zsize || zsize == 0 );
    device_ptr = (data_t*)a.ptr;
    EXCEPTION_ASSERT(elemSize.x*sizeof(data_t)<=pitch);
}

template<typename data_t>
template<typename data_t2>
cudaPitchedPtrType<data_t2> cudaPitchedPtrType<data_t>::reinterpretCast() {
	cudaPitchedPtr a;
	a.ptr = device_ptr;
	a.pitch = pitch;
	a.xsize = elemSize.x*sizeof(data_t);
	a.ysize = elemSize.y;
	return cudaPitchedPtrType<data_t2>(a, elemSize.z);
}

// Special (less generic) implementations for positions of type elemSize3_t
template<>
template<>
inline __device__ void cudaPitchedPtrType<int>::clamp<elemSize3_t>( elemSize3_t& elemPos ) const {
	if (elemPos.x>=elemSize.x) elemPos.x = elemSize.x-1;
	if (elemPos.y>=elemSize.y) elemPos.y = elemSize.y-1;
	if (elemPos.z>=elemSize.z) elemPos.z = elemSize.z-1;
}
template<>
template<>
inline __device__ void cudaPitchedPtrType<unsigned>::clamp<elemSize3_t>( elemSize3_t& elemPos ) const {
	if (elemPos.x>=elemSize.x) elemPos.x = elemSize.x-1;
	if (elemPos.y>=elemSize.y) elemPos.y = elemSize.y-1;
	if (elemPos.z>=elemSize.z) elemPos.z = elemSize.z-1;
}
template<>
template<>
inline __device__ void  cudaPitchedPtrType<unsigned short>::clamp<elemSize3_t>( elemSize3_t& elemPos ) const {
	if (elemPos.x>=elemSize.x) elemPos.x = elemSize.x-1;
	if (elemPos.y>=elemSize.y) elemPos.y = elemSize.y-1;
	if (elemPos.z>=elemSize.z) elemPos.z = elemSize.z-1;
}
template<>
template<>
inline __device__ void cudaPitchedPtrType<float>::clamp<elemSize3_t>( elemSize3_t& elemPos ) const {
        if (elemPos.x>=elemSize.x) elemPos.x = elemSize.x-1;
        if (elemPos.y>=elemSize.y) elemPos.y = elemSize.y-1;
        if (elemPos.z>=elemSize.z) elemPos.z = elemSize.z-1;
}
template<>
template<>
inline __device__ void cudaPitchedPtrType<char4>::clamp<elemSize3_t>( elemSize3_t& elemPos ) const {
	if (elemPos.x>=elemSize.x) elemPos.x = elemSize.x-1;
	if (elemPos.y>=elemSize.y) elemPos.y = elemSize.y-1;
	if (elemPos.z>=elemSize.z) elemPos.z = elemSize.z-1;
}

template<>
template<>
inline __device__ void cudaPitchedPtrType<uchar4>::clamp<elemSize3_t>( elemSize3_t& elemPos ) const {
	if (elemPos.x>=elemSize.x) elemPos.x = elemSize.x-1;
	if (elemPos.y>=elemSize.y) elemPos.y = elemSize.y-1;
	if (elemPos.z>=elemSize.z) elemPos.z = elemSize.z-1;
}

template<>
template<>
inline __device__ void cudaPitchedPtrType<float2>::clamp<elemSize3_t>( elemSize3_t& elemPos ) const {
        if (elemPos.x>=elemSize.x) elemPos.x = elemSize.x-1;
        if (elemPos.y>=elemSize.y) elemPos.y = elemSize.y-1;
        if (elemPos.z>=elemSize.z) elemPos.z = elemSize.z-1;
}

template<>
template<>
inline __device__ void cudaPitchedPtrType<float2>::clamp<uint2>( uint2& elemPos ) const {
        if (elemPos.x>=elemSize.x) elemPos.x = elemSize.x-1;
        if (elemPos.y>=elemSize.y) elemPos.y = elemSize.y-1;
}


template<typename data_t>
template<typename vec_t>
inline __device__ void cudaPitchedPtrType<data_t>::clamp( vec_t& elemPos ) const {
	if (elemPos.x>=elemSize.x) elemPos.x = elemSize.x-1;
	if (elemPos.y>=elemSize.y) elemPos.y = elemSize.y-1;
	if (elemPos.z>=elemSize.z) elemPos.z = elemSize.z-1;

	if (elemPos.x<0) elemPos.x = 0;
	if (elemPos.y<0) elemPos.y = 0;
	if (elemPos.z<0) elemPos.z = 0;
}
