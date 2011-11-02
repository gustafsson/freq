#ifndef WAVELET_CU_H
#define WAVELET_CU_H

#include "datastorage.h"

void        wtInverse( float2* in_wavelet, DataStorage<float, 3>::Ptr out_inverse_waveform, cudaExtent numElem, cudaStream_t stream=0 );

const char* wtGetError();
void        wtCompute( float2* in_waveform_ft, float2* out_wavelet_ft, float sampleRate, float minHz, float maxHz, cudaExtent numElem, unsigned half_sizes, float scales_per_octave, float sigma_t0, float normalization_factor, cudaStream_t stream=0 );
//void        wtInverseCallKernel( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, cudaStream_t stream=0 );
//void        wtInverseEllipse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples, cudaStream_t stream=0 );
//void        wtInverseBox( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples, cudaStream_t stream=0 );
void        wtClamp( cudaPitchedPtrType<float2> in_wt, size_t sample_offset, cudaPitchedPtrType<float2> out_clamped_wt, cudaStream_t stream=0  );
void        stftNormalizeInverse( DataStorage<float, 3>::Ptr wave, unsigned length, cudaStream_t stream=0 );

#endif // WAVELET_CU_H
