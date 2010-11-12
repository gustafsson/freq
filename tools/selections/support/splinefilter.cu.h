#pragma once

#include <cudaPitchedPtrType.h>

void applyspline(
        cudaPitchedPtrType<float2> data,
        cudaPitchedPtrType<float2> spline, bool save_inside );
