#ifndef TEXTUREREADER_CU_H
#define TEXTUREREADER_CU_H

#include "resample.h"
#include "cudaPitchedPtrType.h"


// only supports InputT == float2 and InputT == float for the moment
texture<float2, 1, cudaReadModeElementType> input1_float2;
texture<float2, 2, cudaReadModeElementType> input2_float2;
texture<float, 1, cudaReadModeElementType> input1_float;
texture<float, 2, cudaReadModeElementType> input2_float;
static cudaChannelFormatDesc currentlyBoundFormatDesc;

#define TEXREADCALL static inline

/**
    Texture binding must be made in a static function because the texture
    objects only exists in this compilation pass (because we're using
    the high level NVCC C++ API).

    Thus, we can't bind and unbind the textures in a class constructor/
    destructor because member functions can't be locally linked.
    */
template<typename T>
class Read1D
{
public:
    Read1D( unsigned pitch ) : pitch(pitch) {}

    __device__ T operator()(DataPos const& p);
private:
    unsigned pitch;
};


template<typename T>
class Read2D
{
public:
    __device__ T operator()(DataPos const& p);
};

template<> __device__ inline
float Read2D<float>::
        operator()(DataPos const& p)
{
    return tex2D(input2_float, p.x, p.y);
}

template<> __device__ inline
float2 Read2D<float2>::
        operator()(DataPos const& p)
{
    return tex2D(input2_float2, p.x, p.y);
}


template<> __device__ inline
float Read1D<float>::
        operator()(DataPos const& p)
{
    DataPos q = p;
    if (q.x >= pitch)
        q.x = pitch-1;
    return tex1Dfetch(input1_float, q.x + pitch*q.y);
}


template<> __device__ inline
float2 Read1D<float2>::
        operator()(DataPos const& p)
{
    DataPos q = p;
    if (q.x >= pitch)
        q.x = pitch-1;
    return tex1Dfetch(input1_float2, q.x + pitch*q.y);
}


/**
  Use this function through Read1D_Create or Read2D_Create
  */
template<typename T>
TEXREADCALL void _Read_bindtex( cudaPitchedPtr tex, bool needNeighborhood );

template<>
TEXREADCALL void _Read_bindtex<float>( cudaPitchedPtr tex, bool needNeighborhood )
{
    currentlyBoundFormatDesc = cudaCreateChannelDesc<float>();
    if (needNeighborhood)
    {
        input2_float.addressMode[0] = cudaAddressModeClamp;
        input2_float.addressMode[1] = cudaAddressModeClamp;
        input2_float.filterMode = cudaFilterModePoint;
        input2_float.normalized = false;

        cudaBindTexture2D<float, 2, cudaReadModeElementType>(
                0, input2_float, tex.ptr, currentlyBoundFormatDesc,
                tex.xsize/sizeof(float), tex.ysize, tex.pitch );
    } else {
        input1_float.addressMode[0] = cudaAddressModeClamp;
        input1_float.filterMode = cudaFilterModePoint;
        input1_float.normalized = false;

        cudaBindTexture<float, 1, cudaReadModeElementType>(
                0, input1_float, tex.ptr, currentlyBoundFormatDesc,
                tex.pitch * tex.ysize );
    }
}

template<>
TEXREADCALL void _Read_bindtex<float2>( cudaPitchedPtr tex, bool needNeighborhood )
{
    currentlyBoundFormatDesc = cudaCreateChannelDesc<float2>();
    if (needNeighborhood)
    {
        input2_float2.addressMode[0] = cudaAddressModeClamp;
        input2_float2.addressMode[1] = cudaAddressModeClamp;
        input2_float2.filterMode = cudaFilterModePoint;
        input2_float2.normalized = false;

        cudaBindTexture2D<float2, 2, cudaReadModeElementType>(
                0, input2_float2, tex.ptr, currentlyBoundFormatDesc,
                tex.xsize/sizeof(float2), tex.ysize, tex.pitch );
    } else {
        input1_float2.addressMode[0] = cudaAddressModeClamp;
        input1_float2.filterMode = cudaFilterModePoint;
        input1_float2.normalized = false;

        cudaBindTexture<float2, 1, cudaReadModeElementType>(
                0, input1_float2, tex.ptr, currentlyBoundFormatDesc,
                tex.pitch * tex.ysize );
    }
}


/**
  Use this function through Read2D_Create
  */
template<typename T>
TEXREADCALL void _Read2D_bindtexArray( cudaArray *array );

template<>
TEXREADCALL void _Read2D_bindtexArray<float>( cudaArray *array )
{
    input2_float.addressMode[0] = cudaAddressModeClamp;
    input2_float.addressMode[1] = cudaAddressModeClamp;
    input2_float.filterMode = cudaFilterModePoint;
    input2_float.normalized = false;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
    cudaBindTextureToArray( input2_float, array, desc );
}

template<>
TEXREADCALL void _Read2D_bindtexArray<float2>( cudaArray *array )
{
    input2_float.addressMode[0] = cudaAddressModeClamp;
    input2_float.addressMode[1] = cudaAddressModeClamp;
    input2_float.filterMode = cudaFilterModePoint;
    input2_float.normalized = false;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
    cudaBindTextureToArray( input2_float2, array, desc );
}


template<typename T>
TEXREADCALL Read1D<T> Read1D_Create( cudaPitchedPtrType<T> tex )
{
    cudaPitchedPtr cpp = tex.getCudaPitchedPtr();
    _Read_bindtex<T>( cpp, false );
    return Read1D<T>( cpp.pitch/sizeof(T) );
}

template<typename T>
TEXREADCALL Read2D<T> Read2D_Create( cudaPitchedPtrType<T> tex )
{
    _Read_bindtex<T>( tex.getCudaPitchedPtr(), true );
    return Read2D<T>();
}

template<typename T>
TEXREADCALL Read2D<T> Read2D_Create( cudaArray* array )
{
    _Read2D_bindtexArray<T>( array );
    return Read2D<T>();
}


template<typename T>
TEXREADCALL void Read1D_UnbindTexture();
template<typename T>
TEXREADCALL void Read2D_UnbindTexture();

template<>
TEXREADCALL void Read1D_UnbindTexture<float>()
{
    cudaUnbindTexture( input1_float );
}

template<>
TEXREADCALL void Read1D_UnbindTexture<float2>()
{
    cudaUnbindTexture( input1_float2 );
}

template<>
TEXREADCALL void Read2D_UnbindTexture<float>()
{
    cudaUnbindTexture( input2_float );
}

template<>
TEXREADCALL void Read2D_UnbindTexture<float2>()
{
    cudaUnbindTexture( input2_float2 );
}

#endif // TEXTUREREADER_CU_H
