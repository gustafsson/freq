#ifndef WAVELET_CU_H
#define WAVELET_CU_H

#include "chunkdata.h"

void        wtInverse( Tfr::ChunkData::Ptr in_wavelet, DataStorage<float>::Ptr out_inverse_waveform, DataStorageSize x );

const char* wtGetError();
void        wtCompute( DataStorage<Tfr::ChunkElement>::Ptr in_waveform_ft, Tfr::ChunkData::Ptr out_wavelet_ft, float sampleRate, float minHz, float maxHz, unsigned half_sizes, float scales_per_octave, float sigma_t0, float normalization_factor );
//void        wtInverseCallKernel( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, cudaStream_t stream=0 );
//void        wtInverseEllipse( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples, cudaStream_t stream=0 );
//void        wtInverseBox( float2* in_wavelet, float* out_inverse_waveform, cudaExtent numElem, float4 area, unsigned n_valid_samples, cudaStream_t stream=0 );
void        wtClamp( Tfr::ChunkData::Ptr in_wt, size_t sample_offset, Tfr::ChunkData::Ptr out_clamped_wt );
void        stftNormalizeInverse( DataStorage<float>::Ptr wave, unsigned length );

#endif // WAVELET_CU_H
