#pragma once

#include <cudaPitchedPtrType.h>
#include <cuda_vector_types_op.h>

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


    float2 pos; // mu
    float2 k;
    float scale;

private:
    void normalized_scale(float2 sigma)
    {
        // TODO this should normalize the gaussian, check
        scale = 1.0/(2.0*M_PI*sigma.x*sigma.y);
    }
};

void addGauss(
        float4 imageArea, cudaPitchedPtrType<float> image, Gauss gauss );
void multiplyGauss(
        float4 imageArea, cudaPitchedPtrType<float> image, Gauss gauss );
