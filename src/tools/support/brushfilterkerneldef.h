#ifndef BRUSHFILTERKERNEL_H
#define BRUSHFILTERKERNEL_H

#include "brushfilterkernel.h"

class MultiplyOperator
{
public:
#ifdef __CUDACC__
    __device__ void operator()(Tfr::ChunkElement& e, float const& v)
    {
        float a = exp2f(v);
        float2& f = (float2&)e;
        f = f*a;
    }
#else
    void operator()(Tfr::ChunkElement& e, float const& v)
    {
        float a = exp2f(v);
        e *= a;
    }
#endif
};



void multiply( ResampleArea cwtArea, Tfr::ChunkData::ptr cwt,
               ResampleArea imageArea, DataStorage<float>::ptr image )
{
    resample2d_plain<NoConverter<float>, MultiplyOperator>(
            image,
            cwt,
            imageArea,
            cwtArea,
            false
    );
}

#endif // BRUSHFILTERKERNEL_H
