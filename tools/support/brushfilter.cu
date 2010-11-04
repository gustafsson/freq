#include "brushfilter.cu.h"
#include <resample.cu.h>


class ConvertToFloat2
{
public:
    __device__ float2 operator()(float const& v, uint2 const& )
    {
        return make_float2(v, 0);
    }
};


class MultiplyOperator
{
public:
    __device__ void operator()(float2& e, float2 const& v)
    {
        e.x *= exp2f(v.x);
        e.y *= exp2f(v.x);
    }
};


void multiply( float4 cwtArea, cudaPitchedPtrType<float2> cwt,
               float4 imageArea, cudaPitchedPtrType<float> image )
{
    resample2d_plain<float, float2, ConvertToFloat2, MultiplyOperator>(
            image,
            cwt,
            imageArea,
            cwtArea,
            false
    );
}
