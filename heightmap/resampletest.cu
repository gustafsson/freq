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
            //make_float4(0.1,0.1,0.9,0.893702),
  //          make_float4(0.1,0.1,0.9,0.9),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.1,0.1,0.9,0.9),
            make_float4(0,0,1,1),
            make_float4(0,0,1,1),
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


class CoordinateTestFetcher
{
public:
    template<typename Reader>
    __device__ float operator()( float2 const& p, Reader& reader )
    {
        // Plot "how wrong" the phase is
        float2 q = p;
        q.x /= getWidth(validInputs4)-1;
        q.y /= getHeight(validInputs4)-1;
        q.x *= getWidth(inputRegion);
        q.y *= getHeight(inputRegion);
        q.x += getLeft(inputRegion);
        q.y += getTop(inputRegion);

        return q.y;
    }

    float4 inputRegion;
    uint4 validInputs4;
};


void coordinatetest_resample2d(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float> output
        )
{
    elemSize3_t insz = input.getNumberOfElements();
    elemSize3_t outsz = output.getNumberOfElements();
    float4 inputRegion = make_float4(0,0,1,1);
    float4 outputRegion = make_float4(0,0,1,1);
    uint4 validInputs4 = make_uint4(0,0,insz.x,insz.y);
    uint2 validOutputs = make_uint2(outsz.x, outsz.y);

    CoordinateTestFetcher coordinatetest;
    coordinatetest.inputRegion = inputRegion;
    coordinatetest.validInputs4 = validInputs4;
    resample2d_fetcher<float, float2, float, CoordinateTestFetcher, AssignOperator<float> >(
            input,
            output,
            validInputs4,
            validOutputs,
            inputRegion,
            outputRegion,
            false,
            coordinatetest
    );

    AffineTransform at(
            inputRegion,
            make_float4(0,0.5,1,1),
            validInputs4,
            make_uint2(outsz.x, outsz.y)
            );
    float2 a[] =
    {
        make_float2(0,0),
        make_float2(outsz.x-1,outsz.y-1),
        make_float2(0,outsz.y-1),
        make_float2(outsz.x-1,0)
    };
    for (unsigned i=0; i<sizeof(a)/sizeof(a[0]);++i)
    {
        float2 t = at(a[i]);

        float2 q = t;
        q.x /= getWidth(validInputs4)-1;
        q.y /= getHeight(validInputs4)-1;
        q.x *= getWidth(inputRegion);
        q.y *= getHeight(inputRegion);
        q.x += getLeft(inputRegion);
        q.y += getTop(inputRegion);

        printf("%u: at(%g, %g) = (%g, %g), global (%g, %g)\n", i, a[i].x, a[i].y, t.x, t.y, q.x, q.y-0.5);
    }
}
