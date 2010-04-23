#ifndef FILTER_CU_H
#define FILTER_CU_H

#include <cuda_runtime.h>

void        removeDisc( float2* wavelet, cudaExtent numElem, float4 area );
void        removeRect( float2* wavelet, cudaExtent numElem, float4 area );

#endif // FILTER_CU_H
