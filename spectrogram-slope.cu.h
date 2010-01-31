#ifndef SPECTROGRAMSLOPE_CU_H
#define SPECTROGRAMSLOPE_CU_H

extern "C"
void cudaCalculateSlopeKernel(  float* h, float2 *slopeOut,
                                unsigned int width, unsigned int height);

#endif // SPECTROGRAMSLOPE_CU_H
