#ifndef DRAWNWAVEFORM_CU_H
#define DRAWNWAVEFORM_CU_H

#include "cudaPitchedPtrType.h"

#define drawWaveform_BLOCK_SIZE (32)
#define drawWaveform_YRESOLUTION (1024)

/**
 Plot the waveform on the matrix
 */
void drawWaveform( cudaPitchedPtrType<float> in_waveform,
                   cudaPitchedPtrType<float2> out_waveform_matrix,
                   float blob, unsigned readstop, float maxValue );

#endif // DRAWNWAVEFORM_CU_H
