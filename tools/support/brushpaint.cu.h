#pragma once

#include <cudaPitchedPtrType.h>

void addGauss( float4 imageArea, cudaPitchedPtrType<float> image,
               float2 pos, float2 sigma, float scale );

void multiplyGauss( float4 imageArea, cudaPitchedPtrType<float> image,
               float2 pos, float2 sigma, float scale );
