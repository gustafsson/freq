#include <resample.cu.h>

#include "slope.cu.h"

class SlopeFetcher
{
public:
    SlopeFetcher( float xscale, float yscale )
        :   xscale( xscale ),
            yscale( yscale )
    {

    }


    template<typename Reader>
    __device__ float2 operator()( uint2 const& p, Reader& reader )
    {
        // Rely on reader to do clamping
        int top=-1, left=-1;

        // clamp
        if (p.x == 0)
            left = 0;
        if (p.y == 0)
            top = 0;

        float2 slope = make_float2(
            (reader(make_uint2(p.x + 1, p.y)) - reader(make_uint2(p.x + left, p.y)))*xscale,
            (reader(make_uint2(p.x, p.y+1)) - reader(make_uint2(p.x, p.y+top)))*yscale);

        return slope;
    }

private:
    const float xscale;
    const float yscale;
};


extern "C"
void cudaCalculateSlopeKernel(  cudaPitchedPtrType<float> heightmapIn,
                                cudaPitchedPtrType<float2> slopeOut,
                                float xscale, float yscale )
{
    elemSize3_t sz_input = heightmapIn.getNumberOfElements();
    elemSize3_t sz_output = slopeOut.getNumberOfElements();

    uint4 validInputs = make_uint4( 0, 0, sz_input.x, sz_input.y );
    uint2 validOutputs = make_uint2( sz_output.x, sz_output.y );

    resample2d_fetcher<float2>(heightmapIn, slopeOut,
                               validInputs, validOutputs,
                               make_float4(0,0,1,1),
                               make_float4(0,0,1,1),
                               false,
                               SlopeFetcher( 1/xscale, 10/yscale ),
                               AssignOperator<float2>() );
}
