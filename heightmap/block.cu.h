#ifndef HEIGHTMAPBLOCK_CU_H
#define HEIGHTMAPBLOCK_CU_H

#include <cudaPitchedPtrType.h>
#include "tfr/freqaxis.h"

/**
  The namespace Tfr does not know about the namespace Heightmap
  */
namespace Heightmap {

    // TODO find a better name
    enum ComplexInfo {
        ComplexInfo_Amplitude_Weighted,
        ComplexInfo_Amplitude_Non_Weighted,
        ComplexInfo_Phase
    };
};

extern "C"
void blockMergeChunk( cudaPitchedPtrType<float2> inChunk,
                 cudaPitchedPtrType<float> outBlock,
                 float in_sample_rate,
                 float out_sample_rate,
                 float in_frequency_resolution,
                 float out_frequency_resolution,
                 unsigned in_sample_offset,
                 float out_sample_offset,
                 float in_frequency_offset,
                 float out_frequency_offset,
                 float out_count,
                 Heightmap::ComplexInfo transformMethod,
                 unsigned cuda_stream);

extern "C"
void blockResampleChunk( cudaPitchedPtrType<float2> input,
                 cudaPitchedPtrType<float> output,
                 uint2 validInputs,
                 float4 inputRegion,
                 float4 outputRegion,
                 Heightmap::ComplexInfo transformMethod
                 );

extern "C"
void blockMerge( cudaPitchedPtrType<float> inBlock,
                 cudaPitchedPtrType<float> outBlock,
                 float4 in_area,
                 float4 out_area );

extern "C"
void blockMergeOld( cudaPitchedPtrType<float> inBlock,
                 cudaPitchedPtrType<float> outBlock,
                 float in_sample_rate,
                 float out_sample_rate,
                 float in_frequency_resolution,
                 float out_frequency_resolution,
                 float in_offset,
                 float out_offset,
                 float in_valid_samples,
                 unsigned cuda_stream);

extern "C"
void expandStft( cudaPitchedPtrType<float2> inStft,
                 cudaPitchedPtrType<float> outBlock,
                 float min_hz,
                 float max_hz,
                 float out_offset,
                 float out_length,
                 unsigned cuda_stream);


extern "C"
void expandCompleteStft( cudaPitchedPtrType<float2> inStft,
                 cudaPitchedPtrType<float> outBlock,
                 float out_min_hz,
                 float out_max_hz,
                 float out_stft_size,
                 float in_offset,
                 float in_min_hz,
                 float in_max_hz,
                 unsigned in_stft_size,
                 unsigned cuda_stream);

extern "C"
void resampleStft( cudaPitchedPtrType<float2> input,
                   cudaPitchedPtrType<float> output,
                   float4 inputRegion,
                   float4 outputRegion,
                   Tfr::FreqAxis inputAxis,
                   Tfr::FreqAxis outputAxis );

#endif // HEIGHTMAPBLOCK_CU_H
