#ifndef DRAWNWAVEFORM_CU_H
#define DRAWNWAVEFORM_CU_H

#include "tfr/chunkdata.h"

#define drawWaveform_BLOCK_SIZE (32)
#define drawWaveform_YRESOLUTION (1024)

/**
 Plot the waveform on the matrix
 */
void drawWaveform( DataStorage<float>::Ptr in_waveform,
                   Tfr::ChunkData::Ptr out_waveform_matrix,
                   float blob, int readstop, float maxValue, float writeposoffs );

void drawWaveform( DataStorage<float>::Ptr in_waveform,
                   DataStorage<float>::Ptr out_waveform_matrix,
                   float blob, int readstop, float maxValue, float writeposoffs, float y0 );

#endif // DRAWNWAVEFORM_CU_H
