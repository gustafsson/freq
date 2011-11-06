#include <resamplecuda.cu.h>

#include "slopekerneldef.h"

// that's it, slopekerneldef contains the entry function

/*
extern "C"
void cudaCalculateSlopeKernelArray(  cudaArray* heightmapIn, cudaExtent sz_input,
                                DataStorage<std::complex<float> >::Ptr slopeOutp,
                                float xscale, float yscale )
{
    cudaPitchedPtrType<float2> slopeOut( CudaGlobalStorage::WriteAll<2>( slopeOutp ).getCudaPitchedPtr() );

    elemSize3_t sz_output = slopeOut.getNumberOfElements();

    uint4 validInputs = make_uint4( 0, 0, sz_input.width, sz_input.height );
    uint2 validOutputs = make_uint2( sz_output.x, sz_output.y );

    resample2d_fetcher_array<float2, float>(heightmapIn, slopeOut,
                               validInputs, validOutputs,
                               make_float4(0,0,1,1),
                               make_float4(0,0,1,1),
                               false,
                               SlopeFetcher( 10/xscale, 1/yscale, make_uint2( sz_input.width, sz_input.height) ),
                               AssignOperator<float2>() );
}
*/
