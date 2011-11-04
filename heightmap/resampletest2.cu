#include "resampletest.cu.h"
#include <stdio.h>

#include <resamplecuda.cu.h>

void simple_resample2d_2(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float> output
        )
{
    void (*myptr)(cudaPitchedPtrType<float2>,cudaPitchedPtrType<float>);
    myptr = &simple_resample2d_2;
    printf("&simple_resample2d_2 = %p\n", (void*)myptr);

    resample2d_plain<ConverterAmplitude >(
            input,
            output,
            ResampleArea(0.1, 0.1, 0.9, 0.9),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.0,0.0,0.5,0.5),
            ResampleArea(0, 0, 1, 1),
//            make_float4(0,0,1,1),
            false
    );
}
