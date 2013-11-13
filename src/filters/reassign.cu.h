#ifndef REASSIGN_CU_H
#define REASSIGN_CU_H

#include "tfr/chunkdata.h"

void        tonalizeFilter( Tfr::ChunkData::Ptr chunk, float min_hz, float max_hz, float sample_rate );
void        reassignFilter( Tfr::ChunkData::Ptr chunk, float min_hz, float max_hz, float sample_rate );

#endif // REASSIGN_CU_H
