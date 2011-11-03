#pragma once

#include "heightmap/block.cu.h"

#include <cudaPitchedPtrType.h>
#include <cuda_vector_types_op.h>

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

class Gauss
{
public:
    Gauss(float2 pos, float2 sigma)
        :   pos(pos)
    {
        normalized_scale(sigma);
        k = M_LOG2E*0.5/(sigma*sigma);
    }


    Gauss(float2 pos, float2 sigma, float scale)
        :   pos(pos),
            scale(scale)
    {
        k = M_LOG2E*0.5/(sigma*sigma);
    }


    float gauss_value(float x, float y)
    {
        return gauss_value(make_float2(x,y));
    }


    __host__ __device__ float gauss_value(float2 const& v)
    {
        float2 r = (v - pos);
        r = r*r*k; // k = log2f(e)*0.5f/sigma/sigma
        return scale*exp2f(-r.x-r.y);
    }


    float2 sigma()
    {
        // k = log2f(e)*0.5f/sigma/sigma
        return make_float2(
                sqrtf( M_LOG2E*0.5f/k.x ),
                sqrtf( M_LOG2E*0.5f/k.y )
            );
    }

    float2 pos; // mu
    float2 k; // k = log2f(e)*0.5f/sigma/sigma
    float scale;

private:
    void normalized_scale(float2 sigma)
    {
        // this normalizes the bivariate gaussian
        scale = 1.0/(2.0*M_PI*sigma.x*sigma.y);
    }
};


struct ImageArea
{
    float t1, s1, t2, s2;
};

void addGauss(
        ImageArea imageArea, DataStorage<float>::Ptr image, Gauss gauss );
void multiplyGauss(
        ImageArea imageArea, DataStorage<float>::Ptr image, Gauss gauss, Heightmap::AmplitudeAxis );
