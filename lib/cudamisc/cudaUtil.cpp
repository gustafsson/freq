#include "cudaUtil.h"
#include <stdexcept>


//#ifdef _DEBUG
# include "stringprintf.h"
//#endif // _DEBUG

#define MB *(1<<20)

template<typename A> const A& min(const A& a, const A& b)
{
    return a<b?a:b;
}

template<typename A> const A& max(const A& a, const A& b)
{
    return a>b?a:b;
}


size_t
        availableMemoryForSingleAllocation( )
{
    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);

    size_t margin = 16<<20; // 16 MB margin
    free -= min(free, margin);

    return free;
}


// EstimateFreeMem
// Tries to estimate how big the biggest chunk of GPU memory that can be allocated at this moment is.
// The returned value should be accurate to a degree of a few megabytes.
// Even though accuracy is set to half a megabyte cudaMalloc might fail and succeed in a apparent
// inconsistent way due to other processes running on the graphics card.
unsigned int
EstimateFreeMem( unsigned int accuracy )
{
#ifdef _DEBUG
	cerrprintf("Brutally estimating largest available memory chunk, lots of cudaError_enum exceptions are to follow...\n");
#endif

	unsigned
		cursor = 1 MB,
		step,
		returnValue = 0;
	
    accuracy = max(accuracy, (unsigned)(.1 MB) );

	void*
		ptr;

	// Find a value that is too big
	while(cudaSuccess == cudaMalloc( &ptr, cursor<<=1 ) ) {
		returnValue = cursor;
		cudaFree( ptr );
	}


	// Half of that value is too small
	// Start refining...


	// Decrease step size to half as long as accuracy is smaller
	step = returnValue;//>>1;
	while( (step>>=1) > accuracy>>1 )
		if ( cudaSuccess == cudaMalloc( &ptr, cursor ) ) {
			// Ok, move up
			returnValue = cursor;
			cudaFree( ptr );
			cursor += step;
		}
		else
		{
			// Nope, move down
			cursor -= step;
		}

	cudaGetLastError();

#ifdef _DEBUG
	cerrprintf("Estimated free memory: %u MB (%u B), sorry for the cudaError_enum exceptions above.\n", returnValue>>20, returnValue);
#endif

	return returnValue;
}

inline cudaError_t cudaFreeArrayAndZero( cudaArray** x ) {
	cudaError_t e = cudaSuccess;
	if (x!=0 && *x!=0) {
		e = cudaFreeArray (*x); *x=0;}
	return e;
}
#include <stdlib.h>
void freeAndZero( void** x ) {
	if ((x)!=0 && (*x)!=0) {
		free (*x); (*x)=0;}
}

cudaError_t cudaFreeAndZero( void** x ) {
	cudaError_t e = cudaSuccess;
	if ((x)!=0 && (*x)!=0) {
		e = cudaFree (*x); (*x)=0;}
	return e;
}

uint3 wrapCudaGrid( const ushort3 nElems, const dim3& block ) {
    return wrapCudaGrid( make_uint3(nElems.x, nElems.y, nElems.z), block );
}
uint3 wrapCudaGrid( const uint3 nElems, const dim3& block ) {
        uint3 pitch = {
                int_div_ceil(nElems.x, block.x),
                int_div_ceil(nElems.y, block.y),
                int_div_ceil(nElems.z, block.z)};

	const unsigned maxGrid = 65535;

	if (pitch.x*pitch.y>maxGrid || pitch.z>maxGrid)
        throw std::domain_error(printfstring("wrapCudaGrid: To many elements (%u, %u, %u) to fit in one Cuda batch with block size(%u, %u, %u), resulting pitch(%u, %u, %u).", nElems.x, nElems.y, nElems.z, block.x, block.y, block.z, pitch.x, pitch.y, pitch.z));
	
	uint3 grid = {pitch.x*pitch.y, pitch.z, 1};
	return grid;
}

uint3 wrapCudaMaxGrid( const ushort3 nElems, const dim3& block ) {
    return wrapCudaMaxGrid( make_uint3(nElems.x, nElems.y, nElems.z), block );
}
uint3 wrapCudaMaxGrid( const uint3 nElems, const dim3& block ) {
	uint3 pitch = {
                int_div_ceil(nElems.x, block.x),
                int_div_ceil(nElems.y, block.y),
                int_div_ceil(nElems.z, block.z)};

	unsigned totalNumberOfBlocks = pitch.x
								 * pitch.y
								 * pitch.z;

	uint3 grid = {totalNumberOfBlocks, 1, 1};
	// Just max it
	const unsigned maxGrid = 65535;
	if (grid.x>maxGrid) {
        grid.y = int_div_ceil(grid.x, maxGrid);
		grid.x = maxGrid;
	}

	if (grid.y>maxGrid) {
        throw std::domain_error(printfstring("wrapCudaMaxGrid: To many elements (%u, %u, %u) to fit in one Cuda batch with block size(%u, %u, %u).", nElems.x, nElems.y, nElems.z, block.x, block.y, block.z));
	}

	return grid;
}
