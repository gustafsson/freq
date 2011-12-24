#include <resamplecuda.cu.h>

#include <cuda_vector_types_op.h>

#include "splinefilterkerneldef.h"


void applyspline(
        Tfr::ChunkData::Ptr data,
        DataStorage<Tfr::ChunkElement>::Ptr splinep, bool save_inside, float fs )
{
    cudaPitchedPtrType<float2> spline( CudaGlobalStorage::ReadOnly<1>(splinep).getCudaPitchedPtr());

    // cast from Tfr::ChunkElement to float2
    DataStorage<float2>::Ptr data2 =
            CudaGlobalStorage::BorrowPitchedPtr<float2>(
                    data->size(),
                    CudaGlobalStorage::ReadOnly<2>( data ).getCudaPitchedPtr()
                    );

    Spliner< Read1D<float2>, float2 > spliner(
            Read1D_Create<float2>( spline ),
            spline.getNumberOfElements().x,
            save_inside, 1/fs );

    element_operate<float2>( data2, ResampleArea(0, 0, data2->size().width, data2->size().height), spliner );

    Read1D_UnbindTexture<float2>();
}
