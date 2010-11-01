#ifndef REASSIGN_CU_H
#define REASSIGN_CU_H

#include <cudaPitchedPtrType.h>

void        tonalizeFilter( cudaPitchedPtrType<float2> chunk, float min_hz, float max_hz, float sample_rate );
void        reassignFilter( cudaPitchedPtrType<float2> chunk, float min_hz, float max_hz, float sample_rate );

#endif // REASSIGN_CU_H
