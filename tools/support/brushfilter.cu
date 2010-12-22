#include "brushfilter.cu.h"
#include <resample.cu.h>


/**
  resample2d reads one type and converts it to another type to write.
  ConvertToFloat2 makes resample2d pass only one float to MultiplyOperator
  before assignment, where 2 floats are written.
  */
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
        e.y *= exp2f(v.x); // yes, v.x in both places, see ConvertToFloat2
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
