#ifndef RESAMPLETEST_CU_H
#define RESAMPLETEST_CU_H

#include <cudaPitchedPtrType.h>

void simple_resample2d(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float> output
        );


#endif // RESAMPLETEST_CU_H
