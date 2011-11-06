#ifndef HEIGHTMAPSLOPE_CU_H
#define HEIGHTMAPSLOPE_CU_H

#include "datastorage.h"
#include <complex>

extern "C"
void cudaCalculateSlopeKernel(  DataStorage<float>::Ptr heightmapIn,
                                DataStorage<std::complex<float> >::Ptr slopeOut,
                                float xscale, float yscale );

/*extern "C"
void cudaCalculateSlopeKernelArray( cudaArray* heightmapIn, cudaExtent sz_input,
                                DataStorage<std::complex<float> >::Ptr slopeOut,
                                float xscale, float yscale );*/

#endif // HEIGHTMAPSLOPE_CU_H
