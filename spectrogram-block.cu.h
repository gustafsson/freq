#include <cudaPitchedPtrType.h>

extern "C"
void blockMerge( cudaPitchedPtrType<float> outBlock,
                 cudaPitchedPtrType<float2> inChunk,
                 float in_sample_rate,
                 float out_sample_rate,
                 float in_frequency_resolution,
                 float out_frequency_resolution,
                 float in_offset,
                 float out_offset);
