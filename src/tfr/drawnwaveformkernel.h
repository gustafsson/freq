#ifndef DRAWNWAVEFORM_CU_H
#define DRAWNWAVEFORM_CU_H

#include "tfr/chunkdata.h"

#define drawWaveform_BLOCK_SIZE (32)
#define drawWaveform_YRESOLUTION (1024)

/**
 Plot the waveform on the matrix
 */
void drawWaveform( DataStorage<float>::ptr in_waveform,
                   Tfr::ChunkData::ptr out_waveform_matrix,
                   float blob, int readstop, float maxValue, float writeposoffs );

void drawWaveform( DataStorage<float>::ptr in_waveform,
                   DataStorage<float>::ptr out_waveform_matrix,
                   float blob, int readstop, float maxValue, float writeposoffs, float y0 );

#endif // DRAWNWAVEFORM_CU_H
