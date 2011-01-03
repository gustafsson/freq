#include "brushpaint.cu.h"
#include <operate.cu.h>

class AddGaussOperator
{
public:
    AddGaussOperator( Gauss const& g) :g(g) {}

    __device__ void operator()(float& e, float2 const& v)
    {
        e += g.gauss_value(v);
        if (e>10) e = 10;
        if (e<-10) e = -10;
    }
private:
    Gauss g;
};


void addGauss( float4 imageArea, cudaPitchedPtrType<float> image, Gauss g )
{
    AddGaussOperator gauss(g);

    element_operate<float, AddGaussOperator>(image, imageArea, gauss);
}


class MultiplyGaussOperator
{
public:
    MultiplyGaussOperator( Gauss const& g) :g(g) {}

    __device__ void operator()(float& e, float2 const& v)
    {
        //e *= exp2f(g.gauss_value(v));
        e *= exp2f(g.gauss_value(v)*0.2f); // "*0.2f" because of ConverterLogAmplitude
    }
private:
    Gauss g;
};


void multiplyGauss( float4 imageArea, cudaPitchedPtrType<float> image, Gauss g )
{
    MultiplyGaussOperator gauss(g);

    element_operate<float, MultiplyGaussOperator>(image, imageArea, gauss);
}
