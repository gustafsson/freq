#include "resampletest.cu.h"
#define NORESAMPLE
#ifndef NORESAMPLE
#include "resample.cu.h"
#endif
void simple_resample2d(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float> output
        )
{
#ifndef NORESAMPLE
    resample2d_plain<float2, float,ConverterAmplitude >(
            input,
            output,
            make_float4(0.1,0.1,0.9,0.9),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.0,0.0,0.5,0.5),
            make_float4(0,0,1,1),
//            make_float4(0,0,1,1),
            false
    );
#endif
}
