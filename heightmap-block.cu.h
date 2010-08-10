#ifndef HEIGHTMAPBLOCK_CU_H
#define HEIGHTMAPBLOCK_CU_H

#include <cudaPitchedPtrType.h>

namespace Heightmap {
    enum TransformMethod {
        TransformMethod_Cwt,
        TransformMethod_Cwt_phase,
        TransformMethod_Cwt_reassign,
        TransformMethod_Cwt_ridge,
        TransformMethod_Stft
    };
};

extern "C"
void blockMergeChunk( cudaPitchedPtrType<float2> inChunk,
                 cudaPitchedPtrType<float> outBlock,
                 float in_sample_rate,
                 float out_sample_rate,
                 float in_frequency_resolution,
                 float out_frequency_resolution,
                 unsigned in_offset,
                 float out_offset,
                 unsigned in_count,
                 Heightmap::TransformMethod transformMethod,
                 unsigned cuda_stream);

extern "C"
void blockMerge( cudaPitchedPtrType<float> inBlock,
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
void expandCompleteStft( cudaPitchedPtrType<float> inStft,
                 cudaPitchedPtrType<float> outBlock,
                 float out_min_hz,
                 float out_max_hz,
                 float out_stft_size,
                 float in_offset,
                 float in_min_hz,
                 float in_max_hz,
                 unsigned in_stft_size,
                 unsigned cuda_stream);

#endif // HEIGHTMAPBLOCK_CU_H
