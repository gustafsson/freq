#ifndef CUDATEMPLATES_CU_H
#define CUDATEMPLATES_CU_H

#include "cudaPitchedPtrType.h"

// 'float4 v' describes a region as
// v.x = left
// v.y = top
// v.z = width
// v.w = height

__host__ __device__ inline float& getLeft  (float4& v) { return v.x; }
__host__ __device__ inline float& getTop   (float4& v) { return v.y; }
__host__ __device__ inline float& getRight (float4& v) { return v.z; }
__host__ __device__ inline float& getBottom(float4& v) { return v.w; }
__host__ __device__ inline float  getWidth (float4 const& v) { return v.z-v.x; }
__host__ __device__ inline float  getHeight(float4 const& v) { return v.w-v.y; }
__host__ __device__ inline unsigned& getLeft  (uint4& v) { return v.x; }
__host__ __device__ inline unsigned& getTop   (uint4& v) { return v.y; }
__host__ __device__ inline unsigned& getRight (uint4& v) { return v.z; }
__host__ __device__ inline unsigned& getBottom(uint4& v) { return v.w; }
__host__ __device__ inline unsigned  getWidth (uint4 const& v) { return v.z-v.x; }
__host__ __device__ inline unsigned  getHeight(uint4 const& v) { return v.w-v.y; }

#endif // CUDATEMPLATES_CU_H
