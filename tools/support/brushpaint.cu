#include "brushpaint.cu.h"
#include <operate.cu.h>
#include "cuda_vector_types_op.h"

class Gauss
{
public:
    __device__ float gauss_value(float2 const& v)
    {
        float2 r = (v - pos);
        r = r*r*sigma;
        return scale*exp2f(-r.y-r.x);
    }

    float2 pos;
    float2 sigma;
    float scale;
};

class AddGaussOperator: public Gauss
{
public:
    __device__ void operator()(float& e, float2 const& v)
    {
        e += gauss_value(v);
    }
};


void addGauss( float4 imageArea, cudaPitchedPtrType<float> image,
               float2 pos, float2 sigma, float scale )
{
    AddGaussOperator gauss;
    gauss.pos = pos;
    gauss.sigma = sigma;
    gauss.scale = scale;

    element_operate<float, AddGaussOperator>(image, imageArea, gauss);
}


class MultiplyGaussOperator: public Gauss
{
public:
    __device__ void operator()(float& e, float2 const& v)
    {
        e *= exp2f(gauss_value(v));
    }
};


void multiplyGauss( float4 imageArea, cudaPitchedPtrType<float> image,
               float2 pos, float2 sigma, float scale )
{
    MultiplyGaussOperator gauss;
    gauss.pos = pos;
    gauss.sigma = sigma;
    gauss.scale = scale;

    element_operate<float, MultiplyGaussOperator>(image, imageArea, gauss);
}
