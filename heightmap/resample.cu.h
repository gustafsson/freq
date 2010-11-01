#ifndef RESAMPLE_CU_H
#define RESAMPLE_CU_H

#include <cudaPitchedPtrType.h>
#include <float.h>

// 'float4 v' describes a region as
// v.x = left
// v.y = top
// v.z = width
// v.w = height

__host__ __device__ inline float& getLeft  (float4& v) { return v.x; }
__host__ __device__ inline float& getTop   (float4& v) { return v.y; }
__host__ __device__ inline float& getRight (float4& v) { return v.z; }
__host__ __device__ inline float& getBottom(float4& v) { return v.w; }
__host__ __device__ inline float  getWidth (float4 const& v) { return v.z-v.x; }
__host__ __device__ inline float  getHeight(float4 const& v) { return v.w-v.y; }
__host__ __device__ inline unsigned& getLeft  (uint4& v) { return v.x; }
__host__ __device__ inline unsigned& getTop   (uint4& v) { return v.y; }
__host__ __device__ inline unsigned& getRight (uint4& v) { return v.z; }
__host__ __device__ inline unsigned& getBottom(uint4& v) { return v.w; }
__host__ __device__ inline unsigned  getWidth (uint4 const& v) { return v.z-v.x; }
__host__ __device__ inline unsigned  getHeight(uint4 const& v) { return v.w-v.y; }


template<typename InputT, typename OutputT>
class NoConverter
{
public:
    __device__ OutputT operator()( InputT v, uint2 const& /*dataPosition*/ )
    {
        return v;
    }
};


class ConverterAmplitude
{
public:
    __device__ float operator()( float2 v, uint2 const& /*dataPosition*/ )
    {
        return sqrt(v.x*v.x + v.y*v.y);
    }
};


class AffineTransform
{
public:
    AffineTransform(
            float4 inputRegion,
            float4 outputRegion,
            uint4 validInputs,
            uint2 outputSize
            )
    {
//      x = writePos.x;
//      x = x / (outputSize.x-1);
//      x = x * width(outputRegion) + left(outputRegion);
//      x = x - left(inputRegion);
//      x = x / width(inputRegion);
//      x = x * (width(validInputs)-1);
//      x = x + left(validInputs);
//      readPos.x = x;
//      Express this as one affine transform by:
//      readPos.x = x * scale.x + translation.x;
        // The if clauses takes care of the special case when one of the
        // dimensions is just one element wide
        translation.x = getLeft(validInputs);
        translation.y = getTop(validInputs);
        scale.x = (getWidth(validInputs)-1) / getWidth(inputRegion);
        scale.y = (getHeight(validInputs)-1) / getHeight(inputRegion);
        translation.x += (getLeft(outputRegion) - getLeft(inputRegion))*scale.x;
        translation.y += (getTop(outputRegion) - getTop(inputRegion))*scale.y;
        if (outputSize.x==1) ++outputSize.x;
        if (outputSize.y==1) ++outputSize.y;
        scale.x *= getWidth(outputRegion)/(outputSize.x-1);
        scale.y *= getHeight(outputRegion)/(outputSize.y-1);
    }

    template<typename Vec2>
    __host__ __device__ Vec2 operator()( Vec2 const& p )
    {
        Vec2 q;
        q.x = translation.x + p.x*scale.x;
        q.y = translation.y + p.y*scale.y;
        return q;
    }

private:
    float2 scale;
    float2 translation;
};


class AffineTransformFlip
{
public:
    AffineTransformFlip(
            float4 inputRegion,
            float4 outputRegion,
            uint4 validInputs,
            uint2 outputSize
            )
    {
        //      x = writePos.x;
        //      x = x / (outputSize.x-1);
        //      x = x * width(outputRegion) + left(outputRegion);
        //      y = x;
        //      y = y - top(inputRegion);
        //      y = y / height(inputRegion);
        //      y = y * (height(validInputs)-1);
        //      y = y + top(validInputs);
        //      readPos.y = y;
        //      Express this as one affine transform by:
        //      readPos.y = x * scale.y + translation.y;
        translation.x = getLeft(validInputs);
        translation.y = getTop(validInputs);
        scale.x = (getWidth(validInputs)-1) / getWidth(inputRegion);
        scale.y = (getHeight(validInputs)-1) / getHeight(inputRegion);
        translation.x += (getTop(outputRegion) - getLeft(inputRegion))*scale.x;
        translation.y += (getLeft(outputRegion) - getTop(inputRegion))*scale.y;
        if (outputSize.x==1) ++outputSize.x;
        if (outputSize.y==1) ++outputSize.y;
        scale.x *= getHeight(outputRegion)/(outputSize.y-1);
        scale.y *= getWidth(outputRegion)/(outputSize.x-1);
    }


    __host__ __device__ float2 operator()( float2 const& p )
    {
        return make_float2(
                translation.x + p.y*scale.x,
                translation.y + p.x*scale.y );
    }

private:
    float2 scale;
    float2 translation;
};


// only supports InputT == float2 for the moment
// TODO move these to a separate file.
// or only one file can include this header
// or figure out a way to do it anyway
texture<float2, 1> input1_float2;
texture<float2, 2> input2_float2;

template<typename T>
class Read2D
{
public:
    __device__ T operator()(unsigned x, unsigned y);
};

static __device__ float2 read2D(unsigned x, unsigned y)
{
    return tex2D(input2_float2, x, y);
}
template<> __device__ inline
float2 Read2D<float2>::operator()(unsigned x, unsigned y)
{
    return read2D(x,y);
}

template<typename T>
class Read1D
{
public:
    Read1D(unsigned pitch):pitch(pitch) {}

    __device__ T operator()(unsigned x, unsigned y);
private:
    unsigned pitch;
};

static __device__ float2 read1D(unsigned x, unsigned y, unsigned pitch)
{
    return tex1Dfetch(input1_float2, x + pitch*y);
}

template<> __device__ inline
float2 Read1D<float2>::operator()(unsigned x, unsigned y)
{
    return read1D( x,y,pitch);
}


template<typename T> inline
__device__ T interpolate( T const& a, T const& b, float k )
{
    return (1-k)*a + k*b;
}

template<> inline
__device__ float2 interpolate( float2 const& a, float2 const& b, float k )
{
    return make_float2(
            interpolate( a.x, b.x, k ),
            interpolate( a.y, b.y, k )
            );
}

template<typename T> inline
__device__ T zero()    { return 0; }

template<> inline
__device__ float2 zero() { return make_float2(0,0); }

template<typename OutputT, typename InputT, typename Converter, typename Reader>
__device__ OutputT fetch( unsigned x, unsigned y, Converter& converter, Reader& reader )
{
    return converter( reader( x, y ), make_uint2(x, y ) );
}


template<typename OutputT, typename InputT, typename Converter, typename Reader>
__device__ OutputT fetch( unsigned x, float y, Converter& converter, Reader& reader )
{
    float yb = floor(y);
    float yk = y-yb;
    unsigned yu = (unsigned)yb;
    OutputT a = (yk==1.f) ? zero<OutputT>() : fetch<OutputT, InputT>( x, yu, converter, reader );
    OutputT b = (yk==0.f) ? zero<OutputT>() : fetch<OutputT, InputT>( x, yu+1, converter, reader );

    return interpolate( a, b, yk );
}


template<typename OutputT, typename InputT, typename Converter, typename Reader, typename YType>
__device__ OutputT fetch( float x, YType y, Converter& converter, Reader& reader )
{
    float xb = floor(x);
    float xk = x-xb;
    unsigned xu = (unsigned)xb;
    OutputT a = (xk==1.f) ? zero<OutputT>() : fetch<OutputT, InputT>( xu, y, converter, reader );
    OutputT b = (xk==0.f) ? zero<OutputT>() : fetch<OutputT, InputT>( xu+1, y, converter, reader );

    return interpolate( a, b, xk );
}

// #define MULTISAMPLE

template<typename OutputT, typename InputT, typename Converter, typename Reader, typename YType>
__device__ OutputT getrow( float x, float x1, float x2, YType y, Converter& converter, Reader& reader )
{
    OutputT c;

#ifdef MULTISAMPLE
    if (floor(x2)-ceil(x1) <= 3)
#endif
        // Very few samples in this interval, interpolate and take middle
        c = fetch<OutputT, InputT>( x, y, converter, reader );
#ifdef MULTISAMPLE
    else
    {
        // Not very few samples in this interval, fetch max value

        // if (floor(x1) < x1)
        c = fetch<OutputT, InputT>( x1, y, converter, reader );

        for (unsigned x=ceil(x1); x<=floor(x2); ++x)
            maxassign( c, fetch<OutputT, InputT>( x, y, converter, reader ));

        if (floor(x2) < x2)
            maxassign( c, fetch<OutputT, InputT>( x2, y, converter, reader ));
    }
#endif
    return c;
}


template<typename A, typename B>
__device__ bool isless( A const& a, B const& b)
{
    return a < b;
}

template<> inline
__device__ bool isless( float2 const& a, float2 const& b)
{
    return a.x*a.x + a.y * a.y < b.x*b.x + b.y*b.y;
}

template<typename T>
__device__ void maxassign(T& a, T const& b )
{
    if ( isless(a, b) )
        a = b;
}

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 8

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
        typename InputT,
        typename OutputT,
        typename Converter,
        typename Transform>
__global__ void resample2d_kernel (
        float4 validInputs,
        unsigned inputPitch,
        OutputT* output,
        uint2 outputSize,
        unsigned outputPitch,
        Transform coordinateTransform,
        Converter converter
        )
{
    uint2 writePos;
    writePos.x = blockIdx.x * BLOCKDIM_X + threadIdx.x;
    writePos.y = blockIdx.y * BLOCKDIM_Y + threadIdx.y;

    if (writePos.x>=outputSize.x)
        return;
    if (writePos.y>=outputSize.y)
        return;

#ifndef MULTISAMPLE
    float2 p = make_float2(writePos.x, writePos.y);
    p = coordinateTransform(p);
    if (p.x < getLeft(validInputs)) return;
    if (p.y < getTop(validInputs)) return;
    if (p.x > getRight(validInputs)-1) return;
    if (p.y > getBottom(validInputs)-1) return;
#else
    float2 p1 = make_float2(writePos.x, writePos.y);
    float2 p2 = p1;
    p2.x += 0.5f;
    p2.y += 0.5f;
    p1.x -= 0.5f;
    p1.y -= 0.5f;

    p1 = coordinateTransform(p1);
    p2 = coordinateTransform(p2);

    float2 p = make_float2(
            (p1.x+p2.x)*.5f,
            (p1.y+p2.y)*.5f );

    if (p1.x < getLeft(validInputs)) p1.x = getLeft(validInputs);
    if (p1.y < getTop(validInputs)) p1.y = getTop(validInputs);
    if (p2.x <= getLeft(validInputs)) return;
    if (p2.y <= getTop(validInputs)) return;
    if (p1.x >= getRight(validInputs)-1) return;
    if (p1.y >= getBottom(validInputs)-1) return;
    if (p2.x > getRight(validInputs)-1) p2.x = getRight(validInputs)-1;
    if (p2.y > getBottom(validInputs)-1) p2.y = getBottom(validInputs)-1;
#endif
    OutputT c=0;

    //Read2D<InputT> reader;
    Read1D<InputT> reader(inputPitch);
#ifndef MULTISAMPLE
    c = getrow<OutputT, InputT>( p.x, 0, 0, p.y, converter, reader);
#else
    if (floor(p2.y)-ceil(p1.y) <= 3.f)
        // Very few samples in this interval, interpolate and take middle position
        c = getrow<OutputT, InputT>( p.x, p1.x, p2.x, p.y, converter, reader);
    else
    {
        // Not very few samples in this interval, fetch max value
        //if (floor(p1.y) < p1.y)
        c = getrow<OutputT, InputT>( p.x, p1.x, p2.x, p1.y, converter, reader);

        for (unsigned y=ceil(p1.y); y<=floor(p2.y); ++y)
            maxassign( c, getrow<OutputT, InputT>( p.x, p1.x, p2.x, y, converter, reader) );

        if (floor(p2.y) < p2.y)
            maxassign( c, getrow<OutputT, InputT>( p.x, p1.x, p2.x, p2.y, converter, reader) );
    }
#endif

    unsigned o = outputPitch*writePos.y + writePos.x;
    //output[o] += 0.005f;
    output[o] = c;
}


// todo remove
#include <stdio.h>

template<typename T>
static void bindtex( cudaPitchedPtr tex, bool needNeighborhood );

template<>
static void bindtex<float2>( cudaPitchedPtr tex, bool needNeighborhood )
{
    if (false && needNeighborhood)
    {
        input2_float2.addressMode[0] = cudaAddressModeClamp;
        input2_float2.addressMode[1] = cudaAddressModeClamp;
        input2_float2.filterMode = cudaFilterModePoint;
        input2_float2.normalized = false;

        cudaBindTexture2D(0, input2_float2, tex.ptr,
                        tex.xsize/sizeof(float2), tex.ysize, tex.pitch );
    } else {
        input1_float2.addressMode[0] = cudaAddressModeClamp;
        input1_float2.filterMode = cudaFilterModePoint;
        input1_float2.normalized = false;

        cudaBindTexture(0, input1_float2, tex.ptr,
                        tex.pitch * tex.ysize );
    }
}


template<typename Vec4>
float4 make_float4( Vec4 const& v )
{
    return make_float4( v.x, v.y, v.z, v.w );
}

template<
        typename InputT,
        typename OutputT,
        typename Converter >
void resample2d(
        cudaPitchedPtrType<InputT> input,
        cudaPitchedPtrType<OutputT> output,
        uint4 validInputs,
        uint2 validOutputs,
        float4 inputRegion = make_float4(0,0,1,1),
        float4 outputRegion = make_float4(0,0,1,1),
        bool flip = false,
        Converter converter = Converter(),
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
    printf("\noutput.getNumberOfElements().x = %u", output.getNumberOfElements().x);
    printf("\noutput.getNumberOfElements().y = %u", output.getNumberOfElements().y);
#endif

    // make sure validInputs is smaller than input size
    getRight(validInputs) = min(getRight(validInputs), input.getNumberOfElements().x);
    getBottom(validInputs) = min(getBottom(validInputs), input.getNumberOfElements().y);
    getLeft(validInputs) = min(getRight(validInputs), getLeft(validInputs));
    getTop(validInputs) = min(getBottom(validInputs), getTop(validInputs));

    // make sure validOutputs is smaller than output size
    validOutputs.x = min(validOutputs.x, output.getNumberOfElements().x);
    validOutputs.y = min(validOutputs.y, output.getNumberOfElements().y);

    bindtex<InputT>( input.getCudaPitchedPtr(), flip );

    dim3 block( BLOCKDIM_X, BLOCKDIM_Y, 1 );
    elemSize3_t sz = output.getNumberOfElements();
    dim3 grid(
            int_div_ceil( sz.x, block.x ),
            int_div_ceil( sz.y, block.y ),
            1 );

    unsigned inputPitch = input.getCudaPitchedPtr().pitch/sizeof(InputT);
    unsigned outputPitch = output.getCudaPitchedPtr().pitch/sizeof(OutputT);
    OutputT* outputPtr = output.ptr();
    //cudaMemset(output.ptr(), 0, output.getTotalBytes());

#ifdef resample2d_DEBUG
    printf("\nvalidOutputs.x = %u", validOutputs.x);
    printf("\nvalidOutputs.y = %u", validOutputs.y);
    printf("\ngetLeft(validInputs) = %u", getLeft(validInputs));
    printf("\ngetTop(validInputs) = %u", getTop(validInputs));
    printf("\ngetRight(validInputs) = %u", getRight(validInputs));
    printf("\ngetBottom(validInputs) = %u", getBottom(validInputs));
    printf("\n");
    fflush(stdout);
#endif
    //outputPtr = outputPtr + (getLeft(validOutputs) + getTop(validOutputs)*outputPitch);

    if (!flip)
    {
        resample2d_kernel
                <InputT, OutputT, Converter, AffineTransform>
                <<< grid, block, 0, cuda_stream >>>
        (
                make_float4(validInputs),
                inputPitch,
                outputPtr,
                validOutputs,
                outputPitch,

                AffineTransform(
                        inputRegion,
                        outputRegion,
                        validInputs,
                        validOutputs
                        ),
                converter
        );
    } else {
        resample2d_kernel
                <InputT, OutputT, Converter, AffineTransformFlip>
                <<< grid, block, 0, cuda_stream >>>
        (
                make_float4(validInputs),
                inputPitch,
                outputPtr,
                validOutputs,
                outputPitch,

                AffineTransformFlip(
                        inputRegion,
                        outputRegion,
                        validInputs,
                        validOutputs
                        ),
                converter
        );
    }
}

template<
        typename InputT,
        typename OutputT,
        typename Converter>
void resample2d_plain(
        cudaPitchedPtrType<InputT> input,
        cudaPitchedPtrType<OutputT> output,
        float4 inputRegion = make_float4(0,0,1,1),
        float4 outputRegion = make_float4(0,0,1,1),
        bool flip=false,
        Converter converter = Converter()
        )
{
    elemSize3_t sz_input = input.getNumberOfElements();
    elemSize3_t sz_output = output.getNumberOfElements();

    uint4 validInputs = make_uint4( 0, 0, sz_input.x, sz_input.y );
    uint2 validOutputs = make_uint2( sz_output.x, sz_output.y );

    resample2d<InputT, OutputT, Converter>(
            input,
            output,
            validInputs,
            validOutputs,
            inputRegion,
            outputRegion,
            flip,
            converter );
}

template<
        typename InputT,
        typename OutputT>
void resample2d_overlap(
        cudaPitchedPtrType<InputT> input,
        cudaPitchedPtrType<OutputT> output,
        float4 inputRegion = make_float4(0,0,1,1),
        float4 outputRegion = make_float4(0,0,1,1)
        )
{
    resample2d<InputT, OutputT, NoConverter<InputT, OutputT> >(
            input,
            output,
            inputRegion,
            outputRegion );
}


#endif // RESAMPLE_CU_H
