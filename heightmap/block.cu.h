#ifndef HEIGHTMAPBLOCK_CU_H
#define HEIGHTMAPBLOCK_CU_H

#include "tfr/freqaxis.h"

// gpusmisc
#include <cudaPitchedPtrType.h>

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

    enum AmplitudeAxis {
        AmplitudeAxis_Linear,
        AmplitudeAxis_Logarithmic,
        AmplitudeAxis_5thRoot
    };
};

extern "C"
void blockResampleChunk( cudaPitchedPtrType<float2> input,
                 cudaPitchedPtrType<float> output,
                 uint2 validInputs,
                 float4 inputRegion,
                 float4 outputRegion,
                 Heightmap::ComplexInfo transformMethod,
                 Tfr::FreqAxis inputAxis,
                 Tfr::FreqAxis outputAxis,
                 Heightmap::AmplitudeAxis amplitudeAxis
                 );

extern "C"
void blockMerge( cudaPitchedPtrType<float> inBlock,
                 cudaPitchedPtrType<float> outBlock,
                 float4 in_area,
                 float4 out_area );
/*
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
*/
extern "C"
void resampleStft( cudaPitchedPtrType<float2> input,
                   cudaPitchedPtrType<float> output,
                   float4 inputRegion,
                   float4 outputRegion,
                   Tfr::FreqAxis inputAxis,
                   Tfr::FreqAxis outputAxis,
                   Heightmap::AmplitudeAxis amplitudeAxis );

#endif // HEIGHTMAPBLOCK_CU_H
