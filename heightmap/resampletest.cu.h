#ifndef RESAMPLETEST_CU_H
#define RESAMPLETEST_CU_H

#include <cudaPitchedPtrType.h>

void simple_resample2d(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float> output
        );
void simple_resample2d_2(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float> output
        );
void simple_operate(
        cudaPitchedPtrType<float2> data
        );

#endif // RESAMPLETEST_CU_H
