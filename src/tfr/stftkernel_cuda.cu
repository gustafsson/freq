#include <cudaglobalstorage.h>

#include "stftkernel.h"

#include "stringprintf.h"

#include <stdexcept>

__global__ void kernel_stftNormalizeInverse( cudaPitchedPtrType<float> wave, float v );
__global__ void kernel_stftNormalizeInverse( cudaPitchedPtrType<float2> inwave, cudaPitchedPtrType<float> outwave, float v );
__global__ void kernel_stftToComplex( cudaPitchedPtrType<float> inwave, cudaPitchedPtrType<float2> outwave );
__global__ void kernel_stftDiscardImag( cudaPitchedPtrType<float2> inwave, cudaPitchedPtrType<float> outwave );
__global__ void kernel_cepstrumPrepareCepstra( cudaPitchedPtrType<float2> cepstra, float normalization );

void stftNormalizeInverse(
        DataStorage<float>::Ptr wavep,
        unsigned length )
{
    cudaPitchedPtrType<float> wave(CudaGlobalStorage::ReadWrite<1>( wavep ).getCudaPitchedPtr());

    dim3 block(128);
    dim3 grid = wrapCudaMaxGrid( wave.getNumberOfElements(), block);

    kernel_stftNormalizeInverse<<<grid, block, 0>>>( wave, 1.f/length );
}


__global__ void kernel_stftNormalizeInverse( cudaPitchedPtrType<float> wave, float v )
{
    unsigned n;
    if( !wave.unwrapGlobalThreadNumber3D(n))
        return;

    wave.ptr()[n] *= v;
}

void stftNormalizeInverse(
        Tfr::ChunkData::Ptr inwavep,
        DataStorage<float>::Ptr outwavep,
        unsigned length )
{
    cudaPitchedPtrType<float2> inwave(CudaGlobalStorage::ReadOnly<1>( inwavep ).getCudaPitchedPtr());
    cudaPitchedPtrType<float> outwave(CudaGlobalStorage::WriteAll<1>( outwavep ).getCudaPitchedPtr());

    dim3 block(128);
    dim3 grid = wrapCudaMaxGrid( outwave.getNumberOfElements(), block);

    if(inwave.getNumberOfElements().x!=outwave.getNumberOfElements().x ||
       inwave.getNumberOfElements().y!=outwave.getNumberOfElements().y ||
       inwave.getNumberOfElements().z!=outwave.getNumberOfElements().z)
    {
        throw std::runtime_error(printfstring(
                "stftNormalizeInverse: inwave.getNumberOfElements() != getNumberOfElements.getTotalBytes(), (%u, %u, %u) != (%u, %u, %u)",
                inwave.getNumberOfElements().x, inwave.getNumberOfElements().y, inwave.getNumberOfElements().z,
                outwave.getNumberOfElements().x, outwave.getNumberOfElements().y, outwave.getNumberOfElements().z
                ));
    }

    kernel_stftNormalizeInverse<<<grid, block, 0>>>( inwave, outwave, 1.f/length );
}


__global__ void kernel_stftNormalizeInverse( cudaPitchedPtrType<float2> inwave, cudaPitchedPtrType<float> outwave, float v )
{
    unsigned n;
    if( !outwave.unwrapGlobalThreadNumber3D(n))
        return;

    // pitch doesn't matter as this is given as one big vector only
    outwave.ptr()[n] = inwave.ptr()[n].x * v;
}


void stftToComplex(
        DataStorage<float>::Ptr inwavep,
        Tfr::ChunkData::Ptr outwavep )
{
    cudaPitchedPtrType<float> inwave(CudaGlobalStorage::ReadOnly<1>( inwavep ).getCudaPitchedPtr());
    cudaPitchedPtrType<float2> outwave(CudaGlobalStorage::WriteAll<1>( outwavep ).getCudaPitchedPtr());

    dim3 block(128);
    dim3 grid = wrapCudaMaxGrid( outwave.getNumberOfElements(), block);

    kernel_stftToComplex<<<grid, block, 0>>>( inwave, outwave );
}


__global__ void kernel_stftToComplex( cudaPitchedPtrType<float> inwave, cudaPitchedPtrType<float2> outwave )
{
    unsigned n;
    if( !outwave.unwrapGlobalThreadNumber3D(n))
        return;

    outwave.ptr()[n] = make_float2(inwave.ptr()[n], 0);
}


void cepstrumPrepareCepstra(
        Tfr::ChunkData::Ptr chunk,
        float normalization )
{
    cudaPitchedPtrType<float2> cepstra(CudaGlobalStorage::ReadWrite<1>( chunk ).getCudaPitchedPtr());

    dim3 block(128);
    dim3 grid = wrapCudaMaxGrid( cepstra.getNumberOfElements(), block);

    kernel_cepstrumPrepareCepstra<<<grid, block, 0>>>( cepstra, normalization );
}


__global__ void kernel_cepstrumPrepareCepstra( cudaPitchedPtrType<float2> cepstra, float normalization )
{
    unsigned n;
    if( !cepstra.unwrapGlobalThreadNumber3D(n))
        return;

    float2& d = cepstra.ptr()[n];
    d = make_float2(logf( 0.001f + sqrt(d.x*d.x + d.y*d.y))*normalization, 0);
}


__global__ void kernel_stftAverage( cudaPitchedPtrType<float2> input,
                                    cudaPitchedPtrType<float2> output,
                                    unsigned width, unsigned averaging, float as )
{
    unsigned n;
    if( !output.unwrapGlobalThreadNumber3D(n))
        return;

    unsigned k = n/width;
    unsigned j = n%width;

    float elem = .0f;
    for (unsigned a=0; a<averaging; ++a)
    {
        float2 v = input.ptr()[(k*averaging + a)*width + j];
        elem += hypot(v.x,v.y);
    }
    output.ptr()[n] = make_float2(as*elem, 0.f);
}


void stftAverage(
        Tfr::ChunkData::Ptr input,
        Tfr::ChunkData::Ptr output,
        unsigned scales )
{
    unsigned width = scales;
    unsigned height = output->size().width/scales;
    unsigned input_height = input->size().width/scales;
    unsigned averaging = input_height / height;

    cudaPitchedPtrType<float2> in(CudaGlobalStorage::ReadOnly<1>( input ).getCudaPitchedPtr());
    cudaPitchedPtrType<float2> out(CudaGlobalStorage::WriteAll<1>( output ).getCudaPitchedPtr());

    dim3 block(128);
    dim3 grid = wrapCudaMaxGrid( out.getNumberOfElements(), block);

    kernel_stftAverage<<<grid, block, 0>>>( in, out, width, averaging, 1.f/averaging );
}
