#include "brushfilter.cu.h"
#include <resample.cu.h>
#include "cudaglobalstorage.h"

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
        float a = exp2f(v.x); // yes, don't use v.y, see ConvertToFloat2
        e.x *= a;
        e.y *= a;
    }
};



void multiply( ImageArea cwtia, Tfr::ChunkData::Ptr cwtp,
               ImageArea imageia, DataStorage<float>::Ptr imagep )
{
    float4 cwtArea = make_float4(cwtia.t1, cwtia.s1, cwtia.t2, cwtia.s2);
    float4 imageArea = make_float4(imageia.t1, imageia.s1, imageia.t2, imageia.s2);

    cudaPitchedPtrType<float2> cwt( CudaGlobalStorage::ReadWrite<2>(cwtp).getCudaPitchedPtr() );
    cudaPitchedPtrType<float> image( CudaGlobalStorage::ReadOnly<2>(imagep).getCudaPitchedPtr() );

    resample2d_plain<float, float2, ConvertToFloat2, MultiplyOperator>(
            image,
            cwt,
            imageArea,
            cwtArea,
            false
    );
}
