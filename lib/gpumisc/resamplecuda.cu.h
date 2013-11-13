#ifndef RESAMPLECUDA_CU_H
#define RESAMPLECUDA_CU_H

#define RESAMPLE_CALL __device__
#define RESAMPLE_ANYCALL __host__ __device__

#include "cudaglobalstorage.h"

#include "resample.h"

#include "texturereader.cu.h"
#include "cudatemplates.cu.h"

#include <float.h>

#ifndef BLOCKDIM_X
#define BLOCKDIM_X 16
#endif

#ifndef BLOCKDIM_Y
#define BLOCKDIM_Y 8
#endif

#ifndef RESAMPLE_MAX_THREADS
#define RESAMPLE_MAX_THREADS 0
#endif

#ifndef RESAMPLE_MIN_BLOCKS
#define RESAMPLE_MIN_BLOCKS 1
#endif

/**
  resample2d_kernel resamples an image to another size, and optionally applies
  a conversion (template argument 'Converter') to each element. Upsampling is
  performed by linear interpolation. Downsampling uses max values, with
  linearly interpolated edges.

  @param input
    Input image

  @param output
    Output image

  @param inputRegion
    translation( inputRegion ) = 'affine transformation of entire input image
    region'. inputRegion must be given in the same unit as outputRegion. The
    region is inclusive which means that if the input contains 5 samples and
    covers the region [0,1] samples are defined at points 0, 0.25, 0.5, 0.75
    and 1. So a signal with discrete sample rate 'fs' over the region [0,4]
    seconds should contain exactly '4*fs+1' number of samples.

  @param outputRegion
    outputRegion is an affine transformation of the entire ouput image region.
    Given in the same unit as inputRegion. The intersection of inputRegion and
    outputRegion will be translated to an area of the input image that is
    resampled, converted, and written to output.

    If the intersection is not covered by outputRegion, then some samples in
    output will not be written to. Samples that are crossed by the intersection
    border will also not be written to. It is up to the caller to handle this
    situation.


  @param validInputs
    Valid inputs may be smaller than the given input image. validInputs are
    given in number of samples. inputRegion still refers to the total input
    image size though, validInputs effectively makes the intersection smaller.

  @param validOutputs
    Valid outputs may be smaller than the given output image. validOutputs are
    given in number of samples. outputRegion still refers to the total output
    image size though, validOutputs effectively makes the intersection smaller.

  @param converter
    Converts OutputT to InputT. This conversion may optionally use the position
    where data is read from. See NoConverter for an example.

  @param translation
    Translates intersection position to input read coordinates.
  */
template<
        typename Fetcher,
        typename Transform,
        typename Reader,
        typename Writer>
static __global__ void 
__launch_bounds__(RESAMPLE_MAX_THREADS, RESAMPLE_MIN_BLOCKS)
resample2d_kernel (
        ValidSamples validInputs,
        Fetcher fetcher,
        ValidSamples validOutputs,
        Transform coordinateTransform,
        Reader reader,
        Writer writer
        )
{
    DataPos writePos(
        blockIdx.x * BLOCKDIM_X + threadIdx.x,
        blockIdx.y * BLOCKDIM_Y + threadIdx.y);

    resample2d_elem(
            writePos,
            validInputs,
            fetcher,
            validOutputs,
            coordinateTransform,
            reader,
            writer
    );
}


template<>
inline RESAMPLE_CALL float ConverterAmplitude::
operator()( float2 v, DataPos const& /*dataPosition*/ )
{
    // slightly faster than sqrtf(f) unless '--use_fast_math' is specified
    // to nvcc
    // return f*rsqrtf(f);
    return sqrtf(v.x*v.x + v.y*v.y);
}


template<>
inline RESAMPLE_CALL float2 interpolate( float2 const& a, float2 const& b, float k )
{
    return make_float2(
            interpolate( a.x, b.x, k ),
            interpolate( a.y, b.y, k )
            );
}


template<typename FetchT, typename OutputT, typename Assignment>
static DefaultWriter<OutputT, Assignment, FetchT> DefaultWriterStorage(
        boost::shared_ptr<DataStorage<OutputT> > outputp,
        DataPos validOutputs,
        Assignment assignment)
{
    cudaPitchedPtr outCpp;
    if (validOutputs.x == outputp->size().width && validOutputs.y == outputp->size().height)
        outCpp = CudaGlobalStorage::WriteAll<2>(outputp).getCudaPitchedPtr();
    else
        outCpp = CudaGlobalStorage::ReadWrite<2>(outputp).getCudaPitchedPtr();

    cudaPitchedPtrType<OutputT> output( outCpp );

    unsigned outputPitch = output.getCudaPitchedPtr().pitch/sizeof(OutputT);
    OutputT* outputPtr = output.ptr();

    return DefaultWriter<OutputT, Assignment, FetchT>(outputPtr, outputPitch, validOutputs.y, assignment);
}


// todo remove
#include <stdio.h>


template<
        typename Reader,
        typename Fetcher,
        typename Writer,
        typename Transform>
static void resample2d_storage(
        ValidSamples validInputs,
        Fetcher fetcher,
        ValidSamples validOutputs,
        Transform transform,
        Reader reader,
        Writer writer,
        cudaStream_t cuda_stream = (cudaStream_t)0
        )
{
#ifdef resample2d_DEBUG
    printf("\ngetLeft(validInputs) = %u", getLeft(validInputs));
    printf("\ngetTop(validInputs) = %u", getTop(validInputs));
    printf("\ngetRight(validInputs) = %u", getRight(validInputs));
    printf("\ngetBottom(validInputs) = %u", getBottom(validInputs));
    printf("\ninput.getNumberOfElements().x = %u", input.getNumberOfElements().x);
    printf("\ninput.getNumberOfElements().y = %u", input.getNumberOfElements().y);
#endif

    dim3 block( BLOCKDIM_X, BLOCKDIM_Y, 1 );
    dim3 grid(
            int_div_ceil( validOutputs.right, block.x ),
            int_div_ceil( validOutputs.bottom, block.y ),
            1 );

#ifdef resample2d_DEBUG
    printf("\nvalidOutputs.x = %u", validOutputs.right);
    printf("\nvalidOutputs.y = %u", validOutputs.bottom);
    printf("\ngetLeft(validInputs) = %u", getLeft(validInputs));
    printf("\ngetTop(validInputs) = %u", getTop(validInputs));
    printf("\ngetRight(validInputs) = %u", getRight(validInputs));
    printf("\ngetBottom(validInputs) = %u", getBottom(validInputs));
    printf("\n");
    fflush(stdout);
#endif

    resample2d_kernel
            <<< grid, block, 0, cuda_stream >>>
    (
            validInputs,
            fetcher,
            validOutputs,

            transform,
            reader,
            writer
    );
}


template<
        typename Fetcher,
        typename Writer,
        typename InputT
        >
static void resample2d_reader(
        boost::shared_ptr<DataStorage<InputT> > inputp,
        Writer writer,
        ValidSamples validInputs,
        ValidSamples validOutputs,
        DataPos outputSize,
        ResampleArea inputRegion = ResampleArea(0,0,1,1),
        ResampleArea outputRegion = ResampleArea(0,0,1,1),
        bool transpose = false,
        Fetcher fetcher = Fetcher()
        )
{
    cudaPitchedPtrType<InputT> input( CudaGlobalStorage::ReadOnly<2>(inputp).getCudaPitchedPtr() );

    bool needNeighborhood = transpose;
    // TODO create a degenerate 2D texture if width is greater than
    // cudaDeviceProp.maxTexture2D[0] = 65536
    if (input.getNumberOfElements().x > 65536)
        needNeighborhood = false;

    unsigned inputPitch = input.getCudaPitchedPtr().pitch/sizeof(InputT);
    if (inputPitch % 4)
        needNeighborhood = false;

    if (needNeighborhood)
    {
        resample2d_transform(
                Read2D_Create( input ),
                writer,
                validInputs, validOutputs, outputSize,
                inputRegion, outputRegion,
                transpose,
                fetcher
        );
        Read2D_UnbindTexture<InputT>();
    }
    else
    {
        resample2d_transform(
                Read1D_Create( input ),
                writer,
                validInputs, validOutputs, outputSize,
                inputRegion, outputRegion,
                transpose,
                fetcher
        );
        Read1D_UnbindTexture<InputT>();
    }
}



/*
template<
        typename FetchT,
        typename InputT,
        typename OutputT,
        typename Fetcher,
        typename Assignment>
static void resample2d_fetcher_array(
        cudaArray* input,
        typename DataStorage<OutputT>::Ptr outputp,
        ValidInputs validInputs,
        ValidOutputs validOutputs,
        ResampleArea inputRegion = ResampleArea(0,0,1,1),
        ResampleArea outputRegion = ResampleArea(0,0,1,1),
        bool transpose = false,
        Fetcher fetcher = Fetcher(),
        Assignment assignment = Assignment(),
        cudaStream_t cuda_stream = (cudaStream_t)0
        )
{
    cudaPitchedPtrType<OutputT> output( CudaGlobalStorage::ReadWrite(outputp).getCudaPitchedPtr() );

    unsigned outputPitch = output.getCudaPitchedPtr().pitch/sizeof(OutputT);
    OutputT* outputPtr = output.ptr();

    // make sure validOutputs is smaller than output size
    validOutputs.x = min(validOutputs.x, output.getNumberOfElements().x);
    validOutputs.y = min(validOutputs.y, output.getNumberOfElements().y);

#ifdef resample2d_DEBUG
    printf("\noutput.getNumberOfElements().x = %u", output.getNumberOfElements().x);
    printf("\noutput.getNumberOfElements().y = %u", output.getNumberOfElements().y);
    printf("\noutput.ptr() = %p", output.ptr());
    printf("\ninput.ptr() = %p", input.ptr());
#endif

    resample2d_transform<FetchT>(
            Read2D_Create<InputT>( input ),
            DefaultWriterStorage(
                    outputp, validOutputs, assignment),
            validInputs, validOutputs,
            inputRegion, outputRegion, transpose,
            fetcher,
            cuda_stream );

    Read2D_UnbindTexture<InputT>();
}*/


#endif // RESAMPLECUDA_CU_H
