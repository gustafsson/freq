#pragma once

#include <cudaPitchedPtrType.h>
#include <cuda_vector_types_op.h>

class Gauss
{
public:
    __host__ __device__ float gauss_value(float2 const& v)
    {
        float2 r = (v - pos);
        r = r*r*sigma;
        return scale*exp2f(-r.y-r.x);
    }

    float2 pos;
    float2 sigma;
    float scale;
};

void addGauss(
        float4 imageArea, cudaPitchedPtrType<float> image, Gauss gauss );
void multiplyGauss(
        float4 imageArea, cudaPitchedPtrType<float> image, Gauss gauss );
