#ifndef HEIGHTMAPSLOPE_CU_H
#define HEIGHTMAPSLOPE_CU_H

#include <vector_types.h>

extern "C"
void cudaCalculateSlopeKernel(  float* h, float2 *slopeOut,
                                unsigned int width, unsigned int height, float xscale, unsigned cuda_stream);

#endif // HEIGHTMAPSLOPE_CU_H
