#pragma once

#include <cudaPitchedPtrType.h>

void multiply( float4 cwtArea, cudaPitchedPtrType<float2> cwt,
               float4 imageArea, cudaPitchedPtrType<float> image );
