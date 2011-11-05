#ifndef BRUSHPAINTKERNELDEF_H
#define BRUSHPAINTKERNELDEF_H

#include "brushpaintkernel.h"
#include <operate.h>

class AddGaussOperator
{
public:
    AddGaussOperator( Gauss const& g) :g(g) {}

    RESAMPLE_CALL void operator()(float& e, ResamplePos const& v)
    {
        e += g.gauss_value(v);
        if (e>10) e = 10;
        if (e<-10) e = -10;
    }
private:
    Gauss g;
};


void addGauss( ResampleArea imageArea, DataStorage<float>::Ptr imagep, Gauss g )
{
    AddGaussOperator gauss(g);

    element_operate<float, AddGaussOperator>(imagep, imageArea, gauss);
}


class MultiplyGaussOperator
{
public:
    MultiplyGaussOperator( Gauss const& g) :g(g) {}

    RESAMPLE_CALL void operator()(float& e, ResamplePos const& v)
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

    RESAMPLE_CALL void operator()(float& e, ResamplePos const& v)
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

    RESAMPLE_CALL void operator()(float& e, ResamplePos const& v)
    {
        e *= exp2f(g.gauss_value(v)*0.22f);
    }
private:
    Gauss g;
};


void multiplyGauss( ResampleArea imageArea, DataStorage<float>::Ptr imagep, Gauss g, Heightmap::AmplitudeAxis amplitudeAxis )
{
    switch (amplitudeAxis)
    {
    case Heightmap::AmplitudeAxis_Linear:
        element_operate<float, MultiplyGaussOperator>(imagep, imageArea, MultiplyGaussOperator(g));
        break;
    case Heightmap::AmplitudeAxis_Logarithmic:
        element_operate<float, MultiplyGaussOperatorLog>(imagep, imageArea, MultiplyGaussOperatorLog(g));
        break;
    case Heightmap::AmplitudeAxis_5thRoot:
        element_operate<float, MultiplyGaussOperator5th>(imagep, imageArea, MultiplyGaussOperator5th(g));
        break;
    }
}

#endif // BRUSHPAINTKERNELDEF_H
