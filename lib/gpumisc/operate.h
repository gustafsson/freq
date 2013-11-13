#ifndef OPERATE_CU_H
#define OPERATE_CU_H

#include <float.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "cudatemplates.cu.h"
#include "resamplecuda.cu.h"
#else
#include "resamplecpu.h"
#endif

template<typename T>
class AddConstantTimesDistanceOperator
{
public:
    RESAMPLE_CALL void operator()(T& e, ResamplePos const& v)
    {
        e = e + C * sqrtf(v.x*v.x+v.y*v.y);
    }

    T C;
};


class SimpleAffineTransform
{
public:
    SimpleAffineTransform(
            ResampleArea region,
            DataPos size
            )
    {
        translation.x = region.left;
        translation.y = region.top;
        if (size.x==1) ++size.x;
        if (size.y==1) ++size.y;
        scale.x = region.width() / (size.x-1);
        scale.y = region.height() / (size.y-1);
    }

    template<typename Vec2>
    RESAMPLE_ANYCALL Vec2 operator()( Vec2 const& p )
    {
        Vec2 q;
        q.x = translation.x + p.x*scale.x;
        q.y = translation.y + p.y*scale.y;
        return q;
    }

private:
    ResamplePos scale;
    ResamplePos translation;
};


template<
        typename T,
        typename Transform,
        typename ElementOperator>
inline RESAMPLE_CALL void resample2d_elem (
        DataPos writePos,
        T* data,
        DataPos size,
        unsigned pitch,
        Transform coordinateTransform,
        ElementOperator elementOperator
        )
{
    if (writePos.x>=size.x)
        return;
    if (writePos.y>=size.y)
        return;

    ResamplePos p(writePos.x, writePos.y);
    p = coordinateTransform(p);

    unsigned o = pitch*writePos.y + writePos.x;
    elementOperator(data[o], p);
}


#ifdef __CUDACC__

/**
  */
template<
        typename T,
        typename Transform,
        typename ElementOperator>
static __global__ void operate2d_kernel (
        T* data,
        DataPos size,
        unsigned pitch,
        Transform coordinateTransform,
        ElementOperator elementOperator
        )
{
    DataPos writePos(
            blockIdx.x * BLOCKDIM_X + threadIdx.x,
            blockIdx.y * BLOCKDIM_Y + threadIdx.y);

    resample2d_elem(writePos, data, size, pitch, coordinateTransform, elementOperator);
}


template<
        typename T,
        typename ElementOperator >
static void element_operate(
        boost::shared_ptr<DataStorage<T> > datap,
        ResampleArea region = ResampleArea(0,0,1,1),
        ElementOperator elementOperator = ElementOperator(),
        cudaStream_t cuda_stream = (cudaStream_t)0
        )
{
    cudaPitchedPtrType<T> data( CudaGlobalStorage::ReadWrite<2>( datap ).getCudaPitchedPtr() );

    dim3 block( BLOCKDIM_X, BLOCKDIM_Y, 1 );
    elemSize3_t sz = data.getNumberOfElements();
    dim3 grid(
            int_div_ceil( sz.x, block.x ),
            int_div_ceil( sz.y, block.y ),
            1 );

    unsigned pitch = data.getCudaPitchedPtr().pitch/sizeof(T);
    DataPos valid(sz.x, sz.y);
    T* dataPtr = data.ptr();

    operate2d_kernel
            <T, SimpleAffineTransform, ElementOperator>
            <<< grid, block, 0, cuda_stream >>>
    (
            dataPtr,
            valid,
            pitch,

            SimpleAffineTransform(
                    region,
                    valid
                    ),
            elementOperator
    );
}

#else

template<
        typename T,
        typename ElementOperator >
void element_operate(
        boost::shared_ptr<DataStorage<T> > datap,
        ResampleArea region = ResampleArea(0,0,1,1),
        ElementOperator elementOperator = ElementOperator()
        )
{
    DataPos writePos(0,0);
    T* data = CpuMemoryStorage::ReadWrite<2>( datap ).ptr();
    DataPos size(datap->size().width, datap->size().height);
    for (writePos.y = 0; writePos.y<size.y; ++writePos.y)
    {
        for (writePos.x = 0; writePos.x<size.x; ++writePos.x)
        {
            resample2d_elem(
                    writePos,
                    data,
                    size,
                    size.x,

                    SimpleAffineTransform(
                            region,
                            size
                            ),
                    elementOperator
                    );
        }
    }
}

#endif

#endif // OPERATE_CU_H
