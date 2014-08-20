/**
 A bunch of convenient vector operations to use with the cuda vector types.
 Not always the fastest way of doing things, but convenient nontheless.

 johan.b.gustafsson@gmail.com
 */
#pragma once

#include <math.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_types.h>
#include <driver_functions.h>

typedef unsigned short ushort;

// feel free to add more as needed...
inline __device__ __host__ uchar4 operator+( const uchar4& a, const uchar4& b) { uchar4 c = {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; return c; }
inline __device__ __host__ uchar4 operator-( const uchar4& a, const uchar4& b) { uchar4 c = {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; return c; }
inline __device__ __host__ short4 operator-( const char4& a, const char4& b) { short4 c = {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; return c; }
inline __device__ __host__ float4 operator+( const short4& a, const float& b) { float4 c = {a.x+b, a.y+b, a.z+b, a.w+b}; return c; }
inline __device__ __host__ float4 operator*( const float4& a, const float& b) { float4 c = {a.x*b, a.y*b, a.z*b, a.w*b}; return c; }
inline __device__ __host__ float2 operator*( const float2& a, const float& b) { float2 c = {a.x*b, a.y*b}; return c; }
inline __device__ __host__ float2& operator*=( float2& a, const float& b) { a.x*=b; a.y*=b; return a; }
inline __device__ __host__ float2 operator-( const float2& a, const float2& b) { float2 c = {a.x-b.x, a.y-b.y}; return c; }
inline __device__ __host__ float2 operator+( const float2& a, const float2& b) { float2 c = {a.x+b.x, a.y+b.y}; return c; }
inline __device__ __host__ float2 operator+=( float2& a, const float2& b) { a.x+=b.x; a.y+=b.y; return a; }
inline __device__ __host__ float2 operator*( const float2& a, const float2& b) { float2 c = {a.x*b.x, a.y*b.y}; return c; }
inline __device__ __host__ float2 operator/( const double& a, const float2& b) { float2 c = {a/b.x, a/b.y}; return c; }
inline __device__ __host__ int3 operator-( const int3& a, const int3& b) { int3 c = {a.x-b.x, a.y-b.y, a.z-b.z}; return c; }
inline __device__ __host__ ushort3 operator-( const ushort3& a, const ushort3& b) { ushort3 c = {a.x-b.x, a.y-b.y, a.z-b.z}; return c; }
inline __device__ __host__ uint3 operator-( const uint3& a, const uint3& b) { uint3 c = {a.x-b.x, a.y-b.y, a.z-b.z}; return c; }
inline __device__ __host__ uint3 operator+( const uint3& a, const uint3& b) { uint3 c = {a.x+b.x, a.y+b.y, a.z+b.z}; return c; }
inline __device__ __host__ ushort3 operator+( const ushort3& a, const ushort3& b) { ushort3 c = {a.x+b.x, a.y+b.y, a.z+b.z}; return c; }
inline __device__ __host__ short3 operator+( const short3& a, const short3& b) { short3 c = {a.x+b.x, a.y+b.y, a.z+b.z}; return c; }
inline __device__ __host__ float3 operator+( const float3& a, const float3& b) { float3 c = {a.x+b.x, a.y+b.y, a.z+b.z}; return c; }
inline __device__ __host__ char4 operator+( const char4& a, const char4& b) { char4 c = {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; return c; }
inline __device__ __host__ float4 operator+( const float4& a, const float4& b) { float4 c = {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; return c; }
inline __device__ __host__ float3 operator+( const float3& a, int b) { float3 c = {a.x+b, a.y+b, a.z+b}; return c; }
inline __device__ __host__ float3 operator-( const float3& a, const float3& b) { float3 c = {a.x-b.x, a.y-b.y, a.z-b.z}; return c; }
inline __device__ __host__ float3 operator-( const float3& a, const float& b) { float3 c = {a.x-b, a.y-b, a.z-b}; return c; }
inline __device__ __host__ float3 operator-( const float3& a ) { float3 c = {-a.x, -a.y, -a.z}; return c; }
inline __device__ __host__ float3 operator-( const float3& a, int b) { float3 c = {a.x-b, a.y-b, a.z-b}; return c; }
inline __device__ __host__ float3 operator*( const float3& a, const float& b) { float3 c = {a.x*b, a.y*b, a.z*b}; return c; }
inline __device__ __host__ float4 operator*( const char4& a, const float& b) { float4 c = {a.x*b, a.y*b, a.z*b, a.w*b}; return c; }
//inline __device__ char4 operator*( const char4& a, const float& b) { char4 c = {(char)(a.x*b), (char)(a.y*b), (char)(a.z*b), (char)(a.w*b)}; return c; }
inline __device__ __host__ float3 operator*( const float3& a, const ushort3& b) { float3 c = {a.x*b.x, a.y*b.y, a.z*b.z}; return c; }
inline __device__ __host__ float3 operator/( const float3& a, const float3& b) { float3 c = {a.x/b.x, a.y/b.y, a.z/b.z}; return c; }
inline __device__ __host__ float3 operator/( const float3& a, const float& b) { float3 c = {a.x/b, a.y/b, a.z/b}; return c; }
inline __device__ __host__ float4 operator/( const float4& a, const float& b) { float4 c = {a.x/b, a.y/b, a.z/b, a.w/b}; return c; }
inline __device__ __host__ float3 operator/( const float3& a, const ushort3& b) { float3 c = {a.x/b.x, a.y/b.y, a.z/b.z}; return c; }
inline __device__ __host__ float operator%( const float3& a, const float3& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline __device__ __host__ uchar4 operator*( const uchar4& a, const float& f ) { uchar4 c = {(unsigned char)(a.x*f), (unsigned char)(a.y*f), (unsigned char)(a.z*f), (unsigned char)(a.w*f)}; return c; }
inline __device__ __host__ uint3 operator<<( const uint3& a, const uint3& b) {uint3 c = {a.x<<b.x, a.y<<b.y, a.z<<b.z}; return c; }
inline __device__ __host__ ushort3 operator<<( const ushort3& a, const unsigned short& b) {ushort3 c = {a.x<<b, a.y<<b, a.z<<b}; return c; }
inline __device__ __host__ ushort3 operator<<( const ushort3& a, const ushort3& b) {ushort3 c = {a.x<<b.x, a.y<<b.y, a.z<<b.z}; return c; }
inline __device__ __host__ uint3 operator<<( const uint3& a, const int3& b) {uint3 c = {a.x<<b.x, a.y<<b.y, a.z<<b.z}; return c; }
inline __host__ cudaExtent operator<<( const cudaExtent& a, const size_t& b) { return make_cudaExtent(a.width<<b, a.height<<b, a.depth<<b); }
inline __device__ __host__ uint3 operator>>( const uint3& a, const uint3& b) {uint3 c = {a.x>>b.x, a.y>>b.y, a.z>>b.z}; return c; }
inline __device__ __host__ ushort3 operator>>( const ushort3& a, const unsigned short& b) {ushort3 c = {a.x>>b, a.y>>b, a.z>>b}; return c; }
inline __device__ __host__ ushort3 operator>>( const ushort3& a, const ushort3& b) {ushort3 c = {a.x>>b.x, a.y>>b.y, a.z>>b.z}; return c; }
inline __host__ cudaExtent operator>>( const cudaExtent& a, const size_t& b) { return make_cudaExtent(a.width>>b, a.height>>b, a.depth>>b); }
inline __device__ __host__ uint3 operator>>( const uint3& a, const int3& b) {uint3 c = {a.x>>b.x, a.y>>b.y, a.z>>b.z}; return c; }
inline __device__ __host__ bool operator==( const uint3& a, const uint3& b) {return a.x==b.x&&a.y==b.y&&a.z==b.z;}
inline __device__ __host__ bool operator==( const float3& a, const float3& b) {return a.x==b.x&&a.y==b.y&&a.z==b.z;}
inline __device__ __host__ bool operator==( const ushort3& a, const ushort3& b) {return a.x==b.x&&a.y==b.y&&a.z==b.z;}
inline __device__ __host__ bool operator==( const cudaExtent& a, const cudaExtent& b) {return a.width==b.width&&a.height==b.height&&a.depth==b.depth;}
inline __device__ __host__ bool operator!=( const uint3& a, const uint3& b) { return ! (a==b); }
inline __device__ __host__ bool operator!=( const float3& a, const float3& b) { return ! (a==b); }
inline __device__ __host__ bool operator!=( const ushort3& a, const ushort3& b) { return ! (a==b); }
inline __device__ __host__ bool operator!=( const cudaExtent& a, const cudaExtent& b) { return ! (a==b); }
//inline __device__ __host__ short4 abs( const short4& a ){short4 c; c.x=abs(a.x); c.y=abs(a.y); c.z=abs(a.z); c.w=abs(a.w);return c;}
inline __device__ __host__ int dot3( const int4& a ){ return a.x*a.x+a.y*a.y+a.z*a.z;}
inline __device__ __host__ int dot3( const short4& a ){ return a.x*a.x+a.y*a.y+a.z*a.z;}
inline __device__ __host__ unsigned short dot3( const char4& a ){ return a.x*a.x+a.y*a.y+a.z*a.z;}
inline __device__ __host__ float dot3( const float3& a ){ return a.x*a.x+a.y*a.y+a.z*a.z;}
inline __device__ __host__ float dot4( const float4& a ){ return a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w;}
inline __device__ __host__ int dot3( const char4& a, const char4& b ){ return a.x*b.x+a.y*b.y+a.z*b.z;}
inline __device__ __host__ int dot3( const short4& a, const char4& b ){ return a.x*b.x+a.y*b.y+a.z*b.z;}
inline __device__ __host__ float3 getNormalized( const float3& a ){ float dot = dot3(a); return 0==dot?a:a / sqrt(dot); }
#ifndef __CUDACC__
inline float max1( const float& a, const float& b ){ return a > b ? a:b; }
inline float min1( const float& a, const float& b ){ return a < b ? a:b; }
inline float max3( const float3& a ){ return max1(a.x, max1(a.y, a.z));}
inline float clamp( const float& a ){ return max1(0.f, min1(1.f,a)); }
inline float3 clamp( const float3& a ){ float3 c = {clamp(a.x), clamp(a.y), clamp(a.z)}; return c; }
#endif
inline __device__ __host__ float4 mix(const float4& a, const float4& b, float m) { return make_float4(a.x*(1-m)+b.x*m, a.y*(1-m)+b.y*m, a.z*(1-m)+b.z*m, a.w*(1-m)+b.w*m); }
#define MakeVect2(a) {(a).x,(a).y}
#define MakeVect3(a) {(a).x,(a).y,(a).z}
#define MakeVect4(a) {(a).x,(a).y,(a).z,(a).w}
#define MinusEqual3(a,b) {(a).x-=(b).x,(a).y-=(b).y,(a).z-=(b).z;}
#define MinusEqual4(a,b) {(a).x-=(b).x,(a).y-=(b).y,(a).z-=(b).z,(a).w-=(b).w;}
#define PlusEqual4(a,b) {(a).x+=(b).x,(a).y+=(b).y,(a).z+=(b).z,(a).w+=(b).w;}
