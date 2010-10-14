#ifndef RESAMPLE_CU_H
#define RESAMPLE_CU_H

#include <cudaPitchedPtrType.h>
#include <float.h>

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


class NoTranslation
{
public:
    __device__ float2 operator()( float2 const& p )
    {
        return p;
    }
};


class TranslationFlipXY
{
public:
    __device__ float2 operator()( float2 const& p )
    {
        float2 q = make_float2(p.y, p.x);
        return q;
    }
};



// 'float4 v' describes a region as
// v.x = left
// v.y = top
// v.z = width
// v.w = height

__device__ float& getLeft  (float4& v) { return v.x; }
__device__ float& getTop   (float4& v) { return v.y; }
__device__ float& getRight (float4& v) { return v.z; }
__device__ float& getBottom(float4& v) { return v.w; }
__device__ float  getWidth (float4 const& v) { return v.z-v.x; }
__device__ float  getHeight(float4 const& v) { return v.w-v.y; }
__device__ unsigned& getLeft  (uint4& v) { return v.x; }
__device__ unsigned& getTop   (uint4& v) { return v.y; }
__device__ unsigned& getRight (uint4& v) { return v.z; }
__device__ unsigned& getBottom(uint4& v) { return v.w; }
__device__ unsigned  getWidth (uint4 const& v) { return v.z-v.x; }
__device__ unsigned  getHeight(uint4 const& v) { return v.w-v.y; }

template<typename T>
__device__ T read( unsigned x, unsigned y );

// only supports InputT == float2 for the moment
texture<float2, 2> input_float2;

template<>
__device__ float2 read<float2>( unsigned x, unsigned y )
{
    return tex2D(input_float2, x, y);
}


template<typename T>
__device__ T interpolate( T const& a, T const& b, float k )
{
    return (1-k)*a + k*b;
}

template<>
__device__ float2 interpolate( float2 const& a, float2 const& b, float k )
{
    return make_float2(
            interpolate( a.x, b.x, k ),
            interpolate( a.y, b.y, k )
            );
}

template<typename OutputT, typename InputT, typename Converter>
__device__ OutputT fetch( unsigned x, unsigned y, Converter& converter )
{
    return converter( read<InputT>( x, y ), make_uint2(x, y ) );
}


template<typename OutputT, typename InputT, typename Converter>
__device__ OutputT fetch( unsigned x, float y, Converter& converter )
{
    float yb = floor(y);
    float yk = y-yb;
    unsigned yu = (unsigned)yb;
    OutputT a = fetch<OutputT, InputT>( x, yu, converter);
    OutputT b = fetch<OutputT, InputT>( x, yu+1, converter);

    return interpolate( a, b, yk );
}


template<typename OutputT, typename InputT, typename Converter, typename YType>
__device__ OutputT fetch( float x, YType y, Converter& converter )
{
    float xb = floor(x);
    float xk = x-xb;
    unsigned xu = (unsigned)xb;
    OutputT a = fetch<OutputT, InputT>( xu, y, converter);
    OutputT b = fetch<OutputT, InputT>( xu+1, y, converter);

    return interpolate( a, b, xk );
}


template<typename OutputT, typename InputT, typename Converter, typename YType>
__device__ OutputT getrow( float x1, float x2, YType y, Converter& converter )
{
    OutputT c;
    bool is_valid = false;

    if (x2-x1 == 1 || x2==x1)
        // Exactly the same sample rate, take the first sample
        maxassign( c, fetch<InputT, OutputT>( (unsigned)ceil(x1), y, converter), is_valid );
    else if (floor(x2)-ceil(x1) <= 3)
        // Very few samples in this interval, interpolate and take middle
        maxassign( c, fetch<InputT, OutputT>( (x1+x2)/2, y, converter), is_valid );
    else
    {
        // Not very few samples in this interval, fetch max value

        if (floor(x1) < x1)
            maxassign( c, fetch<OutputT, InputT>( x1, y, converter ), is_valid);

        for (unsigned x=ceil(x1); x<=floor(x2); ++x)
            maxassign( c, fetch<OutputT, InputT>( x, y, converter ), is_valid);

        if (floor(x2) < x2)
            maxassign( c, fetch<OutputT, InputT>( x2, y, converter ), is_valid);
    }
    return c;
}


template<typename A, typename B>
__device__ bool isless( A const& a, B const& b)
{
    return a < b;
}

template<>
__device__ bool isless( float2 const& a, float2 const& b)
{
    return a.x*a.x + a.y * a.y < b.x*b.x + b.y*b.y;
}

template<typename T>
__device__ void maxassign(T& a, T const& b, bool& is_valid)
{
    if ( isless(a, b) || !is_valid)
    {
        is_valid = true;
        a = b;
    }
}


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
    region'. inputRegion must be given in the same unit as outputRegion.

  @param outputRegion
    outputRegion is an affine transformation of the entire ouput image region.
    Given in the same unit as inputRegion. The intersection of inputRegion and
    outputRegion will be translated to an area of the input image that is
    resampled, converted, and written to output.

    If the intersection is not covered by outputRegion, then some samples in
    output will not be written to. Samples that are crossed by the intersection
    border will also not be written to. It is up to the caller to handle this
    situation.

    Which samples to write to is computed by assuming that outputRegion is an
    affine transformation of the entire ouput image.

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
        typename InputTranslation>
__global__ void resample2d_kernel(
        uint3 inputSize,
        cudaPitchedPtrType<OutputT> output,
        float4 inputRegion,
        float4 outputRegion,
        uint4 validInputs,
        uint4 validOutputs,
        Converter converter,
        InputTranslation translation
        )
{
    float4 intersection = outputRegion;
    if (getLeft( intersection ) < getLeft( inputRegion ))
        getLeft( intersection ) = getLeft( inputRegion );
    if (getTop( intersection ) < getTop( inputRegion ))
        getTop( intersection ) = getTop( inputRegion );
    if (getRight( intersection ) > getRight( inputRegion ))
        getRight( intersection ) = getRight( inputRegion );
    if (getBottom( intersection ) > getBottom( inputRegion ))
        getBottom( intersection ) = getBottom( inputRegion );

    // Ok inputRegion. outputRegion och intersection är (0,0,1,1)
    if(0) if(inputRegion.x < 100)
    {
        output.elem( make_uint3( 1,0,0) ) = make_float2( 2+inputRegion.x, 2+inputRegion.y );
        output.elem( make_uint3( 1,1,0) ) = make_float2( 2+inputRegion.z, 2+inputRegion.w );
        output.elem( make_uint3( 2,0,0) ) = make_float2( 2+outputRegion.x, 2+outputRegion.y );
        output.elem( make_uint3( 2,1,0) ) = make_float2( 2+outputRegion.z, 2+outputRegion.w );
        output.elem( make_uint3( 0,2,0) ) = make_float2( 2+intersection.x, 2+intersection.y );
        output.elem( make_uint3( 1,2,0) ) = make_float2( 2+intersection.z, 2+intersection.w );
        return;
    }

    uint3 writePos;
    bool valid = output.unwrapCudaGrid( writePos );
    if (!valid)
        return;

    // ok validOutputs är (0,0,3,3)
    if(0) if(inputRegion.x < 100)
    {
        output.elem( make_uint3( 0,0,0) ) = make_float2( 2+validOutputs.x, 2+validOutputs.y );
        output.elem( make_uint3( 1,0,0) ) = make_float2( 2+validOutputs.z, 2+validOutputs.w );
        output.elem( make_uint3( 0,1,0) ) = make_float2( 2+getLeft( validOutputs ), 2+getTop( validOutputs ) );
        output.elem( make_uint3( 1,1,0) ) = make_float2( 2+getRight( validOutputs ), 2+getBottom( validOutputs ) );
        return;
    }

    if (writePos.x < getLeft( validOutputs ))
        return;
    if (writePos.y < getTop( validOutputs ))
        return;
    if (writePos.x >= getRight( validOutputs ))
        return;
    if (writePos.y >= getBottom( validOutputs ))
        return;

    // ok, All writePos are occupied for 3x3 matrix
    if(0) if(inputRegion.x < 100)
    {
        output.elem( writePos ) = make_float2( 2, 2 );
        return;
    }

    // Translate writePos to intersection coordinates to figure out where to
    // read from
    // (x1, y1, x2, y2) defines the read region
    float
            x1 = (writePos.x+0) / (float)output.getNumberOfElements().x,
            x2 = (writePos.x+1) / (float)output.getNumberOfElements().x,
            y1 = (writePos.y+0) / (float)output.getNumberOfElements().y,
            y2 = (writePos.y+1) / (float)output.getNumberOfElements().y;

    // ok, (x1, y1, x2, y2) defines the write region
    if(0) if(inputRegion.x < 100)
    {
        output.elem( writePos ) = make_float2( x2, y2 );
        return;
    }
    x1 = getLeft( outputRegion ) + getWidth( outputRegion )*x1;
    x2 = getLeft( outputRegion ) + getWidth( outputRegion )*x2;
    y1 = getTop( outputRegion ) + getHeight( outputRegion )*y1;
    y2 = getTop( outputRegion ) + getHeight( outputRegion )*y2;

    // ok, (x1, y1, x2, y2) defines the intersection read region
    if(0) if(inputRegion.x < 100)
    {
        output.elem( writePos ) = make_float2( x1, y1 );
        return;
    }
    // Check if entire read region is within intersection
    if ( x1 < getLeft( intersection ))
        return;
    if ( x2 > getRight( intersection ))
        return;
    if ( y1 < getTop( intersection ))
        return;
    if ( y2 > getBottom( intersection ))
        return;
    // ok within intersection

    float2 p1 = { x1, y1 };
    float2 p2 = { x2, y2 };
    p1 = translation(p1);
    p2 = translation(p2);
    x1 = p1.x < p2.x ? p1.x : p2.x;
    x2 = p1.x < p2.x ? p2.x : p1.x;
    y1 = p1.y < p2.y ? p1.y : p2.y;
    y2 = p1.y < p2.y ? p2.y : p1.y;

    // ok, (x1, y1, x2, y2) defines the translated region
    if(0) if(inputRegion.x < 100)
    {
        output.elem( writePos ) = make_float2( x2, y2 );
        return;
    }

    // Translate read_region from intersection coordinates to image coordinates
    x1 -= getLeft( inputRegion );
    x2 -= getLeft( inputRegion );
    y1 -= getTop( inputRegion );
    y2 -= getTop( inputRegion );
    x1 /= getWidth( inputRegion );
    x2 /= getWidth( inputRegion );
    y1 /= getHeight( inputRegion );
    y2 /= getHeight( inputRegion );
    x1 *= inputSize.x;
    x2 *= inputSize.x;
    y1 *= inputSize.y;
    y2 *= inputSize.y;

    // ok, for output sample (0,0) region is (0, 0, 1, 1)
    if (0) if(inputRegion.x < 100)
    {
        output.elem( writePos ) = make_float2( x2, y2 );
        return;
    }

    if (floor(x1) < getLeft(validInputs))
        return;
    if (floor(y1) < getTop(validInputs))
        return;
    if (ceil(x2) > getRight(validInputs))
        x2 = getRight(validInputs) - 1;
    if (ceil(y2) > getBottom(validInputs))
        y2 = getBottom(validInputs) - 1;

    if (x2 < x1)
        return;
    if (y2 < y1)
        return;

    // read_region describes a square, find the maxima of the input image
    // within this square.
    // The maxima can only be on a sample point or on the border. Border values
    // are defined with linear interpolation.

    OutputT c;
    bool is_valid = false;
    if (y2-y1 == 1 || y2==y1)
        // Exactly the same sample rate, take the first sample
        maxassign( c, getrow<InputT, OutputT>( x1, x2, (unsigned)ceil(y1), converter), is_valid );
    else if (floor(y2)-ceil(y1) <= 3)
        // Very few samples in this interval, interpolate and take middle
        maxassign( c, getrow<InputT, OutputT>( x1, x2, (y1+y2)/2, converter), is_valid );
    else
    {
        // Not very few samples in this interval, fetch max value
        if (floor(y1) < y1)
            maxassign( c, getrow<InputT, OutputT>( x1, x2, y1, converter), is_valid );

        for (unsigned y=ceil(y1); y<=floor(y2); ++y)
            maxassign( c, getrow<InputT, OutputT>( x1, x2, y, converter), is_valid );

        if (floor(y2) < y2)
            maxassign( c, getrow<InputT, OutputT>( x1, x2, y2, converter), is_valid );
    }

    if (is_valid)
        output.elem( writePos ) = c;
}


template<typename T>
void bindtex( cudaPitchedPtrType<T> tex );

template<>
void bindtex<float2>( cudaPitchedPtrType<float2> tex )
{
    input_float2.addressMode[0] = cudaAddressModeClamp;
    input_float2.addressMode[1] = cudaAddressModeClamp;
    input_float2.filterMode = cudaFilterModePoint;
    input_float2.normalized = false;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();

    cudaBindTexture2D(0, input_float2, tex.ptr(), channelDesc,
                      tex.getNumberOfElements().x, tex.getNumberOfElements().y,
                      tex.getCudaPitchedPtr().pitch);
}

#include <stdio.h>

template<
        typename InputT,
        typename OutputT,
        typename Converter,
        typename InputTranslation>
void resample2d(
        cudaPitchedPtrType<InputT> input,
        cudaPitchedPtrType<OutputT> output,
        uint4 validInputs,
        uint4 validOutputs,
        float4 inputRegion = make_float4(0,0,1,1),
        float4 outputRegion = make_float4(0,0,1,1),
        Converter converter = Converter(),
        InputTranslation translation = InputTranslation(),
        cudaStream_t cuda_stream = (cudaStream_t)0
        )
{
    bindtex( input );

    dim3 block( 256, 1, 1 );
    dim3 grid = output.wrapCudaGrid( block );

    // ok block och grid funkar för 3x3
    if (1) {
        printf("\nblock = (%u, %u, %u)", block.x, block.y, block.z);
        printf("\ngrid = (%u, %u, %u)", grid.x, grid.y, grid.z);
        fflush(stdout);
    }

    resample2d_kernel
            <InputT, OutputT, Converter, InputTranslation>
            <<< grid, block, 0, cuda_stream >>>
    (
            input.getNumberOfElements(),
            output,
            inputRegion,
            outputRegion,
            validInputs,
            validOutputs,
            converter,
            translation
    );
}


template<
        typename InputT,
        typename OutputT,
        typename Converter,
        typename InputTranslation>
void resample2d_plain(
        cudaPitchedPtrType<InputT> input,
        cudaPitchedPtrType<OutputT> output,
        float4 inputRegion = make_float4(0,0,1,1),
        float4 outputRegion = make_float4(0,0,1,1),
        Converter converter = Converter(),
        InputTranslation translation = InputTranslation()
        )
{
    elemSize3_t sz_input = input.getNumberOfElements();
    elemSize3_t sz_output = output.getNumberOfElements();

    uint4 validInputs = make_uint4( 0, 0, sz_input.x, sz_input.y );
    uint4 validOutputs = make_uint4( 0, 0, sz_output.x, sz_output.y );

    resample2d<InputT, OutputT, Converter, InputTranslation>(
            input,
            output,
            validInputs,
            validOutputs,
            inputRegion,
            outputRegion,
            converter,
            translation);
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
    resample2d<InputT, OutputT, NoConverter<InputT, OutputT>, NoTranslation>(
            input,
            output,
            inputRegion,
            outputRegion );
}


#endif // RESAMPLE_CU_H
