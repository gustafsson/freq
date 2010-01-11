#ifndef WAVELET_CU_H
#define WAVELET_CU_H

const char* wtGetError();
void        wtCompute( float* in_waveform_ft, float* out_waveform_ft, unsigned sampleRate, float minHz, float maxHz, cudaExtent numElem, cudaStream_t stream=0 );
void        wtInverse( float* in_wavelett_ft, cudaExtent in_numElem, float* out_inverse_waveform, cudaExtent out_numElem, cudaStream_t stream=0  );
void        wtClamp( float* in_wt, cudaExtent in_numElem, float* out_clamped_wt, cudaExtent out_numElem, cudaExtent out_offset, cudaStream_t stream=0  );

#endif // WAVELET_CU_H
