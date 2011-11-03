#include <resample.cu.h>

#include "slope.cu.h"
#include "cudaglobalstorage.h"

class SlopeFetcher
{
public:
    SlopeFetcher( float xscale, float yscale, uint2 size )
        :   xscale( xscale ),
            yscale( yscale ),
            size( size )
    {

    }


    template<typename Reader>
    __device__ float2 operator()( float2 const& q, Reader& reader )
    {
        uint2 p = make_uint2(floor(q.x+.5f), floor(q.y+.5f));

        int up=1, left=-1, down=-1, right=1;

        // clamp
        if (p.x == 0)
            left = 0;
        if (p.y == 0)
            down = 0;
        if (p.x + 1 == size.x)
            right = 0;
        if (p.y + 1 == size.y)
            up = 0;

        float2 slope = make_float2(
            (reader(make_uint2(p.x + right, p.y)) - reader(make_uint2(p.x + left, p.y)))*xscale/(right-left),
            (reader(make_uint2(p.x, p.y+up)) - reader(make_uint2(p.x, p.y+down)))*yscale/(up-down));

        return slope;
    }

private:
    const uint2 size;
    const float xscale;
    const float yscale;
};


extern "C"
void cudaCalculateSlopeKernel(  DataStorage<float>::Ptr heightmapInp,
                                DataStorage<std::complex<float> >::Ptr slopeOutp,
                                float xscale, float yscale )
{
    cudaPitchedPtrType<float2> slopeOut( CudaGlobalStorage::WriteAll<2>( slopeOutp ).getCudaPitchedPtr() );
    cudaPitchedPtrType<float> heightmapIn( CudaGlobalStorage::ReadOnly<2>( heightmapInp ).getCudaPitchedPtr() );

    elemSize3_t sz_input = heightmapIn.getNumberOfElements();
    elemSize3_t sz_output = slopeOut.getNumberOfElements();

    uint4 validInputs = make_uint4( 0, 0, sz_input.x, sz_input.y );
    uint2 validOutputs = make_uint2( sz_output.x, sz_output.y );

    resample2d_fetcher<float2>(heightmapIn, slopeOut,
                               validInputs, validOutputs,
                               make_float4(0,0,1,1),
                               make_float4(0,0,1,1),
                               false,
                               SlopeFetcher( 1000, 1000, make_uint2( sz_input.x, sz_input.y) ),
                               AssignOperator<float2>() );
}


extern "C"
void cudaCalculateSlopeKernelArray(  cudaArray* heightmapIn, cudaExtent sz_input,
                                DataStorage<std::complex<float> >::Ptr slopeOutp,
                                float xscale, float yscale )
{
    cudaPitchedPtrType<float2> slopeOut( CudaGlobalStorage::WriteAll<2>( slopeOutp ).getCudaPitchedPtr() );

    elemSize3_t sz_output = slopeOut.getNumberOfElements();

    uint4 validInputs = make_uint4( 0, 0, sz_input.width, sz_input.height );
    uint2 validOutputs = make_uint2( sz_output.x, sz_output.y );

    resample2d_fetcher_array<float2, float>(heightmapIn, slopeOut,
                               validInputs, validOutputs,
                               make_float4(0,0,1,1),
                               make_float4(0,0,1,1),
                               false,
                               SlopeFetcher( 10/xscale, 1/yscale, make_uint2( sz_input.width, sz_input.height) ),
                               AssignOperator<float2>() );
}
