#ifndef MOVE_CU_H
#define MOVE_CU_H

#include "tfr/chunkdata.h"

void        moveFilter( Tfr::ChunkData::ptr chunk,
                        float df,
                        float min_hz,
                        float max_hz,
                        float sample_rate,
                        unsigned sample_offset
                        );

#endif // MOVE_CU_H
