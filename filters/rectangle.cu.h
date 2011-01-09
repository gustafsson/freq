#ifndef RECTANGLE_CU_H
#define RECTANGLE_CU_H

#include <cuda_runtime.h>

void        removeRect( float2* wavelet, cudaExtent numElem, float4 area, bool save_inside );

#endif // RECTANGLE_CU_H
