#pragma once

#include <cudaPitchedPtrType.h>

void watershed(
        float4 imageArea, float2 startPos,
        cudaPitchedPtrType<float> image,
        cudaPitchedPtrType<float> intermediate );
void watershedApplyIntermediate(
        cudaPitchedPtrType<float> intermediate,
        cudaPitchedPtrType<float> image);
