#include "resampletest.cu.h"
#include <stdio.h>

#include "resamplecuda.cu.h"

void simple_resample2d_2(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float> output
        )
{
    void (*myptr)(cudaPitchedPtrType<float2>,cudaPitchedPtrType<float>);
    myptr = &simple_resample2d_2;
    printf("&simple_resample2d_2 = %p\n", (void*)myptr);

    DataStorage<float2>::Ptr inputp = CudaGlobalStorage::BorrowPitchedPtr<float2>( input.getNumberOfElements(), input.getCudaPitchedPtr() );
    DataStorage<float>::Ptr outputp = CudaGlobalStorage::BorrowPitchedPtr<float>( output.getNumberOfElements(), output.getCudaPitchedPtr() );

    resample2d_plain<ConverterAmplitude >(
            inputp,
            outputp,
            ResampleArea(0.1, 0.1, 0.9, 0.9),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.0,0.0,0.5,0.5),
            ResampleArea(0, 0, 1, 1),
//            make_float4(0,0,1,1),
            false
    );
}
