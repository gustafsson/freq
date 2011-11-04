#include <resamplecuda.cu.h>

#include "brushpaint.cu.h"
#include <operate.cu.h>
#include "cudaglobalstorage.h"

class AddGaussOperator
{
public:
    AddGaussOperator( Gauss const& g) :g(g) {}

    __device__ void operator()(float& e, ResamplePos const& v)
    {
        e += g.gauss_value(v);
        if (e>10) e = 10;
        if (e<-10) e = -10;
    }
private:
    Gauss g;
};


void addGauss( ImageArea a, DataStorage<float>::Ptr imagep, Gauss g )
{
    float4 imageArea = make_float4(a.t1, a.s1, a.t2, a.s2);
    cudaPitchedPtrType<float> image( CudaGlobalStorage::ReadWrite<2>( imagep ).getCudaPitchedPtr());

    AddGaussOperator gauss(g);

    element_operate<float, AddGaussOperator>(image, imageArea, gauss);
}


class MultiplyGaussOperator
{
public:
    MultiplyGaussOperator( Gauss const& g) :g(g) {}

    __device__ void operator()(float& e, ResamplePos const& v)
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

    __device__ void operator()(float& e, ResamplePos const& v)
    {
        // Depends on constants in ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Logarithmic>
        e += 0.02f * g.gauss_value(v) * 0.6f;
    }
private:
    Gauss g;
};


class MultiplyGaussOperator5th
{
public:
    MultiplyGaussOperator5th( Gauss const& g) :g(g) {}

    __device__ void operator()(float& e, ResamplePos const& v)
    {
        e *= exp2f(g.gauss_value(v)*0.22f);
    }
private:
    Gauss g;
};


void multiplyGauss( ImageArea a, DataStorage<float>::Ptr imagep, Gauss g, Heightmap::AmplitudeAxis amplitudeAxis )
{
    float4 imageArea = make_float4(a.t1, a.s1, a.t2, a.s2);
    cudaPitchedPtrType<float> image( CudaGlobalStorage::ReadWrite<2>( imagep ).getCudaPitchedPtr());

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
