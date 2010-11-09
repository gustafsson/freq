#include "resampletest.cu.h"
#include <stdio.h>

#include <resample.cu.h>
#include <operate.cu.h>

void simple_resample2d(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float> output
        )
{
    void (*myptr)(cudaPitchedPtrType<float2>,cudaPitchedPtrType<float>);
    myptr = &simple_resample2d;
    printf("&simple_resample2d = %p\n", (void*)myptr);

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
}


class Add2
{
public:
    __device__ void operator()( float2& e, float2 p )
    {
        e.x += 3+p.x;
        e.y += p.y;
    }
};

void simple_operate(
        cudaPitchedPtrType<float2> data
        )
{
    element_operate<float2, Add2>( data );
}
