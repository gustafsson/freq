#include <cuda_runtime.h>

#include "exceptionassert.h"
#include "tasktimer.h"

cudaError_t cudaMemcpy3Dfix(const struct cudaMemcpy3DParms *param) {
    const cudaMemcpy3DParms& p = *param;
	// Use cudaMemcpy3D for 3D only
	// But it does not handle 2D or 1D copies well

	if (1<p.extent.depth) {
		return cudaMemcpy3D( &p );

	} else if (1<p.extent.height) {
		// 2D copy

		// Arraycopy
		if (0 != p.srcArray && 0 == p.dstArray) {
			return cudaMemcpy2DFromArray(p.dstPtr.ptr, p.dstPtr.pitch, p.srcArray, p.srcPos.x, p.srcPos.y, p.extent.width, p.extent.height, p.kind);

		} else if(0 == p.srcArray && 0 != p.dstArray) {
			return cudaMemcpy2DToArray(p.dstArray, p.dstPos.x, p.dstPos.y, p.srcPtr.ptr, p.srcPtr.pitch, p.extent.width, p.extent.height, p.kind);

		} else if(0 != p.srcArray && 0 != p.dstArray) {
			return cudaMemcpy2DArrayToArray( p.dstArray, p.dstPos.x, p.dstPos.y, p.srcArray, p.srcPos.x, p.srcPos.y, p.extent.width, p.extent.height, p.kind);

		} else {
			return cudaMemcpy2D( p.dstPtr.ptr, p.dstPtr.pitch, p.srcPtr.ptr, p.srcPtr.pitch, p.extent.width, p.extent.height, p.kind );

		}

	} else {
		// 1D copy

        // p.extent.width should not include pitch
        EXCEPTION_ASSERT( p.extent.width == p.dstPtr.xsize );
        EXCEPTION_ASSERT( p.extent.width == p.srcPtr.xsize );

		// Arraycopy
		if (0 != p.srcArray && 0 == p.dstArray) {
			return cudaMemcpyFromArray(p.dstPtr.ptr, p.srcArray, p.srcPos.x, p.srcPos.y, p.extent.width, p.kind);

		} else if(0 == p.srcArray && 0 != p.dstArray) {
			return cudaMemcpyToArray(p.dstArray, p.dstPos.x, p.dstPos.y, p.srcPtr.ptr, p.extent.width, p.kind);

		} else if(0 != p.srcArray && 0 != p.dstArray) {
			return cudaMemcpyArrayToArray(p.dstArray, p.dstPos.x, p.dstPos.y, p.srcArray, p.srcPos.x, p.srcPos.y, p.extent.width, p.kind);

        } else {
            return cudaMemcpy( p.dstPtr.ptr, p.srcPtr.ptr, p.extent.width, p.kind );

		}
	}
}
