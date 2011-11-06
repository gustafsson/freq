#include "resampletest.cu.h"
#include <stdio.h>

#include <resamplecuda.cu.h>
#include <operate.h>

void simple_resample2d(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float> output
        )
{
    void (*myptr)(cudaPitchedPtrType<float2>,cudaPitchedPtrType<float>);
    myptr = &simple_resample2d;
    printf("&simple_resample2d = %p\n", (void*)myptr);

    DataStorage<float2>::Ptr inputp = CudaGlobalStorage::BorrowPitchedPtr<float2>( input.getNumberOfElements(), input.getCudaPitchedPtr() );
    DataStorage<float>::Ptr outputp = CudaGlobalStorage::BorrowPitchedPtr<float>( output.getNumberOfElements(), output.getCudaPitchedPtr() );

    resample2d_plain<ConverterAmplitude>(
            inputp,
            outputp,
            //make_float4(0.1,0.1,0.9,0.893702),
  //          make_float4(0.1,0.1,0.9,0.9),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.0,0.0,0.5,0.5),
//            make_float4(0.1,0.1,0.9,0.9),
            ResampleArea(0,0,1,1),
            ResampleArea(0,0,1,1),
            false
    );
}


class Add2
{
public:
    __device__ void operator()( float2& e, ResamplePos p )
    {
        e.x += 3+p.x;
        e.y += p.y;
    }
};

void simple_operate(
        cudaPitchedPtrType<float2> data
        )
{
    DataStorage<float2>::Ptr inputp = CudaGlobalStorage::BorrowPitchedPtr<float2>( data.getNumberOfElements(), data.getCudaPitchedPtr() );
    element_operate<float2, Add2>( inputp );
}


class CoordinateTestFetcher
{
public:
    typedef float T;

    template<typename Reader>
    __device__ float operator()( ResamplePos const& p, Reader& /*reader*/ )
    {
        float2 q = make_float2(p.x, p.y);
        q.x /= validInputs4.width()-1;
        q.y /= validInputs4.height()-1;
        q.x *= inputRegion.width();
        q.y *= inputRegion.height();
        q.x += inputRegion.left;
        q.y += inputRegion.top;

        return q.y;
    }

    ResampleArea inputRegion;
    ValidInputs validInputs4;
};


void coordinatetest_resample2d(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float> output
        )
{
    elemSize3_t insz = input.getNumberOfElements();
    elemSize3_t outsz = output.getNumberOfElements();
    ResampleArea inputRegion(0,0,1,1);
    ResampleArea outputRegion(0,0,1,1);
    ValidInputs validInputs4(0,0,insz.x,insz.y);
    ValidOutputs validOutputs(outsz.x, outsz.y);

    CoordinateTestFetcher coordinatetest;
    coordinatetest.inputRegion = inputRegion;
    coordinatetest.validInputs4 = validInputs4;

    DataStorage<float2>::Ptr inputp = CudaGlobalStorage::BorrowPitchedPtr<float2>( input.getNumberOfElements(), input.getCudaPitchedPtr() );
    DataStorage<float>::Ptr outputp = CudaGlobalStorage::BorrowPitchedPtr<float>( output.getNumberOfElements(), output.getCudaPitchedPtr() );

    resample2d_fetcher<CoordinateTestFetcher, AssignOperator<float> >(
            inputp,
            outputp,
            validInputs4,
            validOutputs,
            inputRegion,
            outputRegion,
            false,
            coordinatetest
    );

    AffineTransform at(
            inputRegion,
            ResampleArea(0,0.5,1,1),
            validInputs4,
            validOutputs
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
        q.x /= validInputs4.width()-1;
        q.y /= validInputs4.height()-1;
        q.x *= inputRegion.width();
        q.y *= inputRegion.height();
        q.x += inputRegion.left;
        q.y += inputRegion.top;

        printf("%u: at(%g, %g) = (%g, %g), global (%g, %g)\n", i, a[i].x, a[i].y, t.x, t.y, q.x, q.y-0.5);
    }
}
