#include "brushfilter.cu.h"
#include <heightmap/resample.cu.h>


class ConvertToFloat2
{
    float2 operator()(float const& v)
    {
        return make_float2(v, v);
    }
};


class MultiplyOperator
{
    void operator()(float2 e, float2 const& v)
    {
        e.x *= v.x;
        e.y *= v.y;
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
