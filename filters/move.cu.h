#ifndef MOVE_CU_H
#define MOVE_CU_H

#include <cudaPitchedPtrType.h>

void        moveFilter( cudaPitchedPtrType<float2> chunk,
                        float df,
                        float min_hz,
                        float max_hz,
                        float sample_rate,
                        unsigned sample_offset
                        );

#endif // MOVE_CU_H
