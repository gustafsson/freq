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
        e *= exp2f(g.gauss_value(v));
    }
private:
    Gauss g;
};


class MultiplyGaussOperatorLog
{
public:
    MultiplyGaussOperatorLog( Gauss const& g) :g(g) {}

    __device__ void operator()(float& e, float2 const& v)
    {
        // same constant as in ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Logarithmic>
        e += 0.02f * g.gauss_value(v);
    }
private:
    Gauss g;
};


class MultiplyGaussOperator5th
{
public:
    MultiplyGaussOperator5th( Gauss const& g) :g(g) {}

    __device__ void operator()(float& e, float2 const& v)
    {
        e *= exp2f(g.gauss_value(v)*0.2f);
    }
private:
    Gauss g;
};


void multiplyGauss( float4 imageArea, cudaPitchedPtrType<float> image, Gauss g, Heightmap::AmplitudeAxis amplitudeAxis )
{
    switch (amplitudeAxis)
    {
    case Heightmap::AmplitudeAxis_Linear:
        element_operate<float, MultiplyGaussOperator>(image, imageArea, MultiplyGaussOperator(g));
        break;
    case Heightmap::AmplitudeAxis_Logarithmic:
        element_operate<float, MultiplyGaussOperatorLog>(image, imageArea, MultiplyGaussOperatorLog(g));
        break;
    case Heightmap::AmplitudeAxis_5thRoot:
        element_operate<float, MultiplyGaussOperator5th>(image, imageArea, MultiplyGaussOperator5th(g));
        break;
    }
}
