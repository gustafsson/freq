#ifndef ELLIPS_CU_H
#define ELLIPS_CU_H

#include <cuda_runtime.h>

void        removeDisc( float2* wavelet, cudaExtent numElem, float4 area, bool _save_inside );

#endif // ELLIPS_CU_H
