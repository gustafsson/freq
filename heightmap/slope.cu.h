#ifndef HEIGHTMAPSLOPE_CU_H
#define HEIGHTMAPSLOPE_CU_H

#include <cudaPitchedPtrType.h>

extern "C"
void cudaCalculateSlopeKernel(  cudaPitchedPtrType<float> heightmapIn,
                                cudaPitchedPtrType<float2> slopeOut,
                                float xscale, float yscale );

#endif // HEIGHTMAPSLOPE_CU_H
