#pragma once

#include <cuda_runtime.h>

size_t availableMemoryForSingleAllocation();

unsigned int
EstimateFreeMem( unsigned int accuracy = 262144 );

void freeAndZero( void** x );
cudaError_t cudaFreeAndZero( void** x );
cudaError_t cudaFreeArrayAndZero( cudaArray** x );

/**
	Computing volumes deeper than 64 units is not natively supported in cuda.
	So a 3d-grid has to be rewraped to a 2d-grid.
	<p>
	Stack slices of blocks on top of eachother. Let the computation be designed
	for a cGrid = {cGrid.x, cGrid.y, cGrid.z} = volumeSize/block, for some 
	block = {xyz}. Then the actual grid for cuda must be cudaGrid = {cGrid.x, 
	cGrid.y*cGrid.z, 1}, with the same block size. The wrapping 
	cGrid -> cudaGrid is done by 'leastUpperBoundForGrid' before the kernel 
	launch. The wrapping cudaGrid -> cGrid in the kernel is done by 
	getValidThreadPos.
	<p>
	If cGrid.x>65535 or cGrid.y*cGrid.z>65535 (cuda implementation specific 
	constant) this function (leastUpperBoundForGrid) will throw a 
	domain_error exception.
	<p>
	maxCudaGrid, for the same reason as wrapToCudaGrid but with a weaker
	precondition that must be fulfilled: cGrid.x*cGrid.y*cGrid.z65535 <= 65535.
	<p>
	Both wrapCudaGrid and wrapCudaMaxGrid finds a least upper bound within their
	respective restrictions.
*/
uint3 wrapCudaGrid( const ushort3 nElems, const dim3& block );
uint3 wrapCudaMaxGrid( const ushort3 nElems, const dim3& block );
uint3 wrapCudaGrid( const uint3 nElems, const dim3& block );
uint3 wrapCudaMaxGrid( const uint3 nElems, const dim3& block );

#include "cudaPitchedPtrType.h"
