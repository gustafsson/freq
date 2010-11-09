#include "brushpaint.cu.h"
#include <operate.cu.h>

class AddGaussOperator: public Gauss
{
public:
    Gauss g;
    __device__ void operator()(float& e, float2 const& v)
    {
        e += g.gauss_value(v);
    }
};


void addGauss( float4 imageArea, cudaPitchedPtrType<float> image, Gauss g )
{
    AddGaussOperator gauss;
    gauss.g = g;

    element_operate<float, AddGaussOperator>(image, imageArea, gauss);
}


class MultiplyGaussOperator: public Gauss
{
public:
    Gauss g;
    __device__ void operator()(float& e, float2 const& v)
    {
        e *= exp2f(g.gauss_value(v));
    }
};


void multiplyGauss( float4 imageArea, cudaPitchedPtrType<float> image, Gauss g )
{
    MultiplyGaussOperator gauss;
    gauss.g = g;

    element_operate<float, MultiplyGaussOperator>(image, imageArea, gauss);
}
