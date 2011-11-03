#include <stdio.h>

#define FREQAXIS_CALL __device__ __host__
#include "tfr/freqaxis.h"

#include "block.cu.h"
#include <resample.cu.h>
#include "cudaglobalstorage.h"

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#define M_PIf ((float)M_PI)

class ConverterPhase
{
public:
    __device__ float operator()( float2 v, uint2 const& /*dataPosition*/ )
    {
        return atan2(v.y, v.x);
    }
};


class ConverterLogAmplitude
{
public:
    __device__ float operator()( float2 v, uint2 const& dataPosition )
    {
        return log2f(0.0001f + ConverterAmplitude()(v,dataPosition)) - log2f(0.0001f);
    }
};


class Converter5thRootAmplitude
{
public:
    __device__ float operator()( float2 v, uint2 const& /*dataPosition*/ )
    {
        return 0.4f*powf(v.x*v.x + v.y*v.y, 0.1);
    }
};


template<unsigned>
class ConverterAmplitudeAxis
{
public:
/*    ConverterAmplitudeAxis(Heightmap::AmplitudeAxis amplitudeAxis)
        : _amplitudeAxis(amplitudeAxis)
    {

    }
*/
    __device__ float operator()( float2 v, uint2 const& dataPosition );
/*    {
        switch(_amplitudeAxis)
        {
        case Heightmap::AmplitudeAxis_Linear:
            return 25.f * ConverterAmplitude()( v, dataPosition );
        case Heightmap::AmplitudeAxis_Logarithmic:
            return 0.02f * ConverterLogAmplitude()( v, dataPosition );
        case Heightmap::AmplitudeAxis_5thRoot:
            return Converter5thRootAmplitude()( v, dataPosition );
        default:
            return -1.f;
        }
    }
    */
private:
    //Heightmap::AmplitudeAxis _amplitudeAxis;
};


template<>
class ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Linear>
{
public:
    __device__ float operator()( float2 v, uint2 const& dataPosition )
    {
        return 25.f * ConverterAmplitude()( v, dataPosition );
    }
};

template<>
class ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Logarithmic>
{
public:
    __device__ float operator()( float2 v, uint2 const& dataPosition )
    {
        return 0.02f * ConverterLogAmplitude()( v, dataPosition );
    }
};

template<>
class ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_5thRoot>
{
public:
    __device__ float operator()( float2 v, uint2 const& dataPosition )
    {
        return Converter5thRootAmplitude()( v, dataPosition );
    }
};

template<typename DefaultConverter>
class AxisFetcher
{
public:
    AxisFetcher(const DefaultConverter& default_converter)
        :   defaultConverter(default_converter)
    {

    }


    template<typename OtherConverter>
    AxisFetcher& operator=(const AxisFetcher<OtherConverter>& b)
    {
        inputAxis = b.inputAxis;
        outputAxis = b.outputAxis;
        scale = b.scale;
        offs = b.offs;
        return *this;
    }


    template<typename Reader>
    __device__ float operator()( float2 const& p, Reader& reader )
    {
        return (*this)(p, reader, defaultConverter);
    }


    __device__ bool near(float a, float b)
    {
        return a > b*(1.f-1e-2f) && a < b*(1.f+1e-2f);
    }


    template<typename Reader, typename Converter>
    __device__ float operator()( float2 const& p, Reader& reader, const Converter& c = Converter() )
    {
        float2 q;
        // exp2f (called in 'getFrequency') is only 4 multiplies for arch 1.x
        // so these are fairly cheap operations. One reciprocal called in
        // 'getFrequencyScalar' is just as fast.
        q.x = p.x;
        q.y = p.y*scale+offs;
        float hz = outputAxis.getFrequency( q.y );
        q.y = inputAxis.getFrequencyScalar( hz );

        float r = InterpolateFetcher<float, Converter>(c)( q, reader );
        return r;
    }

    Tfr::FreqAxis inputAxis;
    Tfr::FreqAxis outputAxis;
    DefaultConverter defaultConverter;

    float scale;
    float offs;
};


template<typename DefaultConverter>
class AxisFetcherTranspose
{
public:
public:
    AxisFetcherTranspose(const DefaultConverter& default_converter)
        :   defaultConverter(default_converter)
    {

    }

    template<typename Reader>
    __device__ float operator()( float2 const& p, Reader& reader )
    {
        float2 q;
        // exp2f (called in 'getFrequency') is only 4 multiplies for arch 1.x
        // so these are fairly cheap operations. One reciprocal called in
        // 'getFrequencyScalar' is just as fast.
        // Tests have shown that this doen't affect the total execution time,
        // when outputAxis and inputAxis are affine transformations of eachother.
        // Hence the kernel is memory bound.
        float hz = outputAxis.getFrequency( p.x );
        q.x = inputAxis.getFrequencyScalar( hz );
        q.y = p.y;

        float r = InterpolateFetcher<float, DefaultConverter>(defaultConverter)( q, reader );
        return r*factor;
    }

    DefaultConverter defaultConverter;
    Tfr::FreqAxis inputAxis;
    Tfr::FreqAxis outputAxis;
    float factor;
};


template<typename DefaultConverter>
class WeightFetcher
{
public:
    WeightFetcher(const DefaultConverter& b)
        : axes(b)
    {

    }

    template<typename Reader>
    __device__ float operator()( float2 const& p, Reader& reader )
    {
        float v = axes( p, reader );
        float phase1 = axes( p, reader, ConverterPhase() );
        float phase2 = axes( make_float2(p.x, p.y+1), reader, ConverterPhase() );
        float phasediff = phase2 - phase1;
        if (phasediff < -M_PIf ) phasediff += 2*M_PIf;
        if (phasediff > M_PIf ) phasediff -= 2*M_PIf;

        float k = exp2f(-fabsf(phasediff));

        return v * k;
    }

    AxisFetcher<DefaultConverter> axes;
};


class SpecialPhaseFetcher
{
public:
    template<typename Reader>
    __device__ float operator()( float2 const& p, Reader& reader )
    {
        // Plot "how wrong" the phase is
        float2 q1 = p;
        float2 p2 = p;
        p2.x++;
        float2 q2 = p2;
        q1.x /= getWidth(validInputs4)-1;
        q1.y /= getHeight(validInputs4)-1;
        q1.x *= getWidth(inputRegion);
        q1.y *= getHeight(inputRegion);
        q1.x += getLeft(inputRegion);
        q1.y += getTop(inputRegion);
        q2.x /= getWidth(validInputs4)-1;
        q2.y /= getHeight(validInputs4)-1;
        q2.x *= getWidth(inputRegion);
        q2.y *= getHeight(inputRegion);
        q2.x += getLeft(inputRegion);
        q2.y += getTop(inputRegion);

        float phase = InterpolateFetcher<float, ConverterPhase>()( p, reader );
        float phase2 = InterpolateFetcher<float, ConverterPhase>()( p2, reader );
        float v = InterpolateFetcher<float, ConverterLogAmplitude>()( p, reader );
        float phasediff = phase2 - phase;
        if (phasediff < -M_PIf ) phasediff += 2*M_PIf;
        if (phasediff > M_PIf ) phasediff -= 2*M_PIf;

        phasediff = (phasediff);
        float f = freqAxis.getFrequency( q1.y );
        f *= (q2.x - q1.x)*2*M_PIf;
        return f*v;

        /*float expected_phase = f * q.x * 2*M_PIf;
        float phasediff = fmodf( expected_phase+2*M_PIf-phase, 2*M_PIf );

        return expected_phase;*/
        //return freqAxis.getFrequency( q.y )/22050.0f;
        //return /*v **/ phasediff / (2*M_PIf);
    }

    Tfr::FreqAxis freqAxis;
    float4 inputRegion;
    uint4 validInputs4;
};


//    cuda-memcheck complains even on this testkernel when using global memory
//    from OpenGL but not on cudaMalloc'd memory. See MappedVbo test.
//__global__ void testkernel(
//        float* output, elemSize3_t sz)
//{
//    elemSize3_t  writePos;
//    writePos.x = blockIdx.x * 128 + threadIdx.x;
//    writePos.y = blockIdx.y * 1 + threadIdx.y;
//    if (writePos.x<sz.x && writePos.y < sz.y)
//    {
//        unsigned o = writePos.x  +  writePos.y * sz.x;
//        o = o % 32;
//        output[o] = 0;
//    }
//}

#include <iostream>
using namespace std;


template<typename AxisConverter>
void blockResampleChunkAxis( cudaPitchedPtrType<float2> input,
                 cudaPitchedPtrType<float> output,
                 uint2 validInputs, // validInputs is the first and last-1 valid samples in x
                 float4 inputRegion,
                 float4 outputRegion,
                 Heightmap::ComplexInfo transformMethod,
                 Tfr::FreqAxis inputAxis,
                 Tfr::FreqAxis outputAxis,
                 AxisConverter amplitudeAxis
                 )
{
    elemSize3_t sz_input = input.getNumberOfElements();
    elemSize3_t sz_output = output.getNumberOfElements();

//    cuda-memcheck complains even on this testkernel when using global memory
//    from OpenGL but not on cudaMalloc'd memory. See MappedVbo test.
//    dim3 block( 128 );
//    dim3 grid( int_div_ceil( sz_output.x, block.x ), sz_output.y );
//    testkernel<<< grid, block>>>(output.ptr(), sz_output);
//    float*devptr;
//    cudaMalloc(&devptr, sizeof(float)*32);
//    testkernel<<< grid, block>>>(devptr, sz_output);
//    cudaFree(devptr);
//    return;

    // We wan't to do our own translation from coordinate position to matrix
    // position in the input matrix. By giving bottom=0 and top=2 we tell
    // 'resample2d_fetcher' to only translate to a normalized reading position
    // [0, height-1) along the input y-axis.
//    uint4 validInputs4 = make_uint4( validInputs.x, 0, validInputs.y, sz_input.y );
    uint4 validInputs4 = make_uint4( validInputs.x, 0, validInputs.y, 2 );
    uint2 validOutputs = make_uint2( sz_output.x, sz_output.y );

    AxisFetcher<AxisConverter> axes = amplitudeAxis;
    axes.inputAxis = inputAxis;
    axes.outputAxis = outputAxis;
    axes.offs = getTop(inputRegion);
    axes.scale = getHeight(inputRegion);

    switch (transformMethod)
    {
    case Heightmap::ComplexInfo_Amplitude_Weighted:
    {
        WeightFetcher<AxisConverter> fetcher = WeightFetcher<AxisConverter>(amplitudeAxis);
        fetcher.axes = axes;

        resample2d_fetcher<float, float2, float, WeightFetcher<AxisConverter>, AssignOperator<float> >(
                input,
                output,
                validInputs4,
                validOutputs,
                inputRegion,
                outputRegion,
                false,
                fetcher
        );
        break;
    }
    case Heightmap::ComplexInfo_Amplitude_Non_Weighted:
        resample2d_fetcher<float, float2, float, AxisFetcher<AxisConverter>, AssignOperator<float> >(
                input,
                output,
                validInputs4,
                validOutputs,
                inputRegion,
                outputRegion,
                false,
                axes
        );
        break;
    case Heightmap::ComplexInfo_Phase:
    {
        AxisFetcher<ConverterPhase> axesPhase = ConverterPhase();
        axesPhase = axes;

        resample2d_fetcher<float, float2, float, AxisFetcher<ConverterPhase>, AssignOperator<float> >(
                    input,
                    output,
                    validInputs4,
                    validOutputs,
                    inputRegion,
                    outputRegion,
                    false,
                    axesPhase
            );
    }
    /*case Heightmap::ComplexInfo_Phase:
            SpecialPhaseFetcher specialPhase;
            specialPhase.freqAxis = freqAxis;
            specialPhase.inputRegion = inputRegion;
            specialPhase.validInputs4 = validInputs4;
            resample2d_fetcher<float, float2, float, SpecialPhaseFetcher, AssignOperator<float> >(
                    input,
                    output,
                    validInputs4,
                    validOutputs,
                    inputRegion,
                    outputRegion,
                    false,
                    specialPhase
            );*/
    }
}



void blockResampleChunk( Tfr::ChunkData::Ptr inputp,
                  BlockData::Ptr outputp,
                 ValidInputs validInput, // validInputs is the first and last-1 valid samples in x
                 BlockArea ia,
                 BlockArea oa,
                 Heightmap::ComplexInfo transformMethod,
                 Tfr::FreqAxis inputAxis,
                 Tfr::FreqAxis outputAxis,
                 Heightmap::AmplitudeAxis amplitudeAxis
                 )
{
    cudaPitchedPtrType<float2> input( CudaGlobalStorage::ReadOnly<2>( inputp ).getCudaPitchedPtr());
    cudaPitchedPtrType<float> output( CudaGlobalStorage::ReadWrite<2>( outputp ).getCudaPitchedPtr());
    float4 inputRegion = make_float4( ia.x1, ia.y1, ia.x2, ia.y2 );
    float4 outputRegion = make_float4( oa.x1, oa.y1, oa.x2, oa.y2 );
    uint2 validInputs = make_uint2(validInput.width, validInput.height);

    switch(amplitudeAxis)
    {
    case Heightmap::AmplitudeAxis_Linear:
        blockResampleChunkAxis(
                input, output, validInputs, inputRegion,
                outputRegion, transformMethod, inputAxis, outputAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Linear>());
        break;
    case Heightmap::AmplitudeAxis_Logarithmic:
        blockResampleChunkAxis(
                input, output, validInputs, inputRegion,
                outputRegion, transformMethod, inputAxis, outputAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Logarithmic>());
        break;
    case Heightmap::AmplitudeAxis_5thRoot:
        blockResampleChunkAxis(
                input, output, validInputs, inputRegion,
                outputRegion, transformMethod, inputAxis, outputAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_5thRoot>());
        break;
    }
}


template<typename AxisConverter>
void resampleStftAxis( Tfr::ChunkData::Ptr inputp,
                   size_t nScales, size_t nSamples,
                   BlockData::Ptr outputp,
                   float4 inputRegion,
                   float4 outputRegion,
                   Tfr::FreqAxis inputAxis,
                   Tfr::FreqAxis outputAxis,
                   Heightmap::AmplitudeAxis amplitudeAxis,
                   AxisConverter axisConverter )
{
    cudaPitchedPtr cpp = CudaGlobalStorage::ReadOnly<2>( inputp ).getCudaPitchedPtr();
    cpp.xsize = cpp.pitch = nScales * sizeof(float2);
    cpp.ysize = nSamples;

    cudaPitchedPtrType<float2> input( cpp );
    cudaPitchedPtrType<float> output( CudaGlobalStorage::ReadWrite<2>( outputp ).getCudaPitchedPtr());

    elemSize3_t sz_input = input.getNumberOfElements();
    elemSize3_t sz_output = output.getNumberOfElements();

    // We wan't to do our own translation from coordinate position to matrix
    // position in the input matrix. By giving left=0 and right=2 we tell
    // 'resample2d_fetcher' to only translate to a normalized reading position
    // [0, width-1) along the input x-axis. 'resample2d_fetcher' does the
    // transpose for us.
    uint4 validInputs = make_uint4( 0, 0, 2, sz_input.y );
    uint2 validOutputs = make_uint2( sz_output.x, sz_output.y );

    AxisFetcherTranspose<AxisConverter> fetcher = axisConverter;
    fetcher.inputAxis = inputAxis;
    fetcher.outputAxis = outputAxis;

    // makes it roughly equal height to Cwt
    switch(amplitudeAxis)
    {
    case Heightmap::AmplitudeAxis_Linear:       fetcher.factor = 0.00052f; break;
    case Heightmap::AmplitudeAxis_Logarithmic:  fetcher.factor = 0.3f; break;
    case Heightmap::AmplitudeAxis_5thRoot:      fetcher.factor = 0.22f; break;
    }

    resample2d_fetcher<float>(
                input,
                output,
                validInputs,
                validOutputs,
                inputRegion,
                outputRegion,
                true,
                fetcher,
                AssignOperator<float>()
        );
}



void resampleStft( Tfr::ChunkData::Ptr input,
                   size_t nScales, size_t nSamples,
                   BlockData::Ptr output,
                   BlockArea ia,
                   BlockArea oa,
                   Tfr::FreqAxis inputAxis,
                   Tfr::FreqAxis outputAxis,
                   Heightmap::AmplitudeAxis amplitudeAxis )
{
    float4 inputRegion = make_float4( ia.x1, ia.y1, ia.x2, ia.y2 );
    float4 outputRegion = make_float4( oa.x1, oa.y1, oa.x2, oa.y2 );

    // fetcher.factor makes it roughly equal height to Cwt
    switch(amplitudeAxis)
    {
    case Heightmap::AmplitudeAxis_Linear:
        resampleStftAxis(
                input, nScales, nSamples, output, inputRegion, outputRegion,
                inputAxis, outputAxis, amplitudeAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Linear>());
        break;
    case Heightmap::AmplitudeAxis_Logarithmic:
        resampleStftAxis(
                input, nScales, nSamples, output, inputRegion, outputRegion,
                inputAxis, outputAxis, amplitudeAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Logarithmic>());
        break;
    case Heightmap::AmplitudeAxis_5thRoot:
        resampleStftAxis(
                input, nScales, nSamples, output, inputRegion, outputRegion,
                inputAxis, outputAxis, amplitudeAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_5thRoot>());
        break;
    }
}


extern "C"
void blockMerge( BlockData::Ptr inBlockp,
                 BlockData::Ptr outBlockp,
                 BlockArea ia,
                 BlockArea oa)
{
    float4 in_area = make_float4( ia.x1, ia.y1, ia.x2, ia.y2 );
    float4 out_area = make_float4( oa.x1, oa.y1, oa.x2, oa.y2 );

    cudaPitchedPtrType<float> inBlock(CudaGlobalStorage::ReadOnly<2>( inBlockp ).getCudaPitchedPtr());
    cudaPitchedPtrType<float> outBlock(CudaGlobalStorage::ReadWrite<2>( outBlockp ).getCudaPitchedPtr());

    resample2d_plain<float, float, NoConverter<float,float> >
            (inBlock, outBlock, in_area, out_area);
}

__global__ void kernel_expand_stft(
                cudaPitchedPtrType<float2> inStft,
                cudaPitchedPtrType<float> outBlock,
                float start,
                float steplogsize,
                float out_offset,
                float out_length )
{
    // Element number
    const unsigned
            y = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned nFrequencies = outBlock.getNumberOfElements().y;
    if( y >= nFrequencies )
        return;

    float ff = y/(float)nFrequencies;
    float hz_out = start*exp(ff*steplogsize);

    float max_stft_hz = 44100.f/2;
    float min_stft_hz = 44100.f/(2*inStft.getNumberOfElements().x);
    float read_f = max(0.f,min(1.f,(hz_out-min_stft_hz)/(max_stft_hz-min_stft_hz)));

    float2 c;

    float p = read_f*inStft.getNumberOfElements().x;
    elemSize3_t readPos = make_elemSize3_t( p, 0, 0 );
    inStft.clamp(readPos);
    c = inStft.elem(readPos);
    float val1 = sqrt(c.x*c.x + c.y*c.y);

    readPos.x++;
    inStft.clamp(readPos);
    c = inStft.elem(readPos);
    float val2 = sqrt(c.x*c.x + c.y*c.y);

    p-=(unsigned)p;
    float val = .02f*(val1*(1-p)+val2*p);
    const float f0 = 2.0f + 35*ff*ff*ff;
    val*=f0;

    elemSize3_t writePos = make_elemSize3_t( 0, y, 0 );
    for (writePos.x=out_offset; writePos.x<out_offset + out_length && writePos.x<outBlock.getNumberOfElements().x;writePos.x++)
    {
        outBlock.e( writePos ) = val;
    }
}


extern "C"
void expandStft( cudaPitchedPtrType<float2> inStft,
                 cudaPitchedPtrType<float> outBlock,
                 float min_hz,
                 float max_hz,
                 float out_offset,
                 float out_length,
                 unsigned cuda_stream)
{
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(outBlock.getNumberOfElements().y, block.x), 1, 1);

    if(grid.x>65535) {
        printf("====================\nInvalid argument, number of floats in complex signal must be less than 65535*256.\n===================\n");
        return;
    }

    float start = min_hz/2;
    float steplogsize = log(max_hz)-log(min_hz);

    kernel_expand_stft<<<grid, block, cuda_stream>>>(
        inStft, outBlock,
        start,
        steplogsize,
        out_offset,
        out_length );
}


// TODO optimize this reading/writing pattern
__global__ void kernel_expand_complete_stft(
                cudaPitchedPtrType<float2> inStft,
                cudaPitchedPtrType<float> outBlock,
                float start,
                float steplogsize,
                float out_stft_size,
                float out_offset,
                float in_min_hz,
                float in_max_hz,
                unsigned in_stft_size)
{
    // Element number
    const unsigned
            x = blockIdx.x*blockDim.x + threadIdx.x,
            y = blockIdx.y*blockDim.y + threadIdx.y;

    float val;
    /*if (1 || 0==threadIdx.x)*/ {
        unsigned nFrequencies = outBlock.getNumberOfElements().y-1;

        // abort if this thread would have written outside outBlock
        if( y > nFrequencies )
            return;

        // which frequency should this thread write
        float ff = y/(float)nFrequencies;
        float hz_write = start*exp(ff*steplogsize);

        // which timestep column should this thread write
        float ts_write = x + out_offset;

        // which normalized frequency should we start reading from
        float hz_read_norm = 0.5f * saturate( (hz_write - in_min_hz)/(in_max_hz - in_min_hz) );

        // which timestep column should we start reading from
        float ts_read = ts_write / out_stft_size;

        if ( 0 > ts_read )
            // only happens if 0>out_offse (or if out_stft_size is negative which is an error)
            return;

        // Compute read coordinates
        // q and p measures how bad read_start is an approximation to ts_read
        // and hz_read_norm
        float q = ts_read - 0.5f;
        float p = max(0.f, min( hz_read_norm*in_stft_size + 0.5f, in_stft_size-1.f ));

        unsigned ts_start = 0 > q ? (unsigned)-1 : (unsigned)q;
        unsigned hz_start = (unsigned)p;
        q -= floor(q);
        p -= hz_start;

        // if the next timestep column is required to compute this outBlock
        // pixel don't compute it unless the next timestep column is provided
        if (0 < q && ts_start+1>=inStft.getNumberOfElements().y)
            return;

        // if a previous timestep column is before the first column, use 0
        // instead

        // if the next or previous frequency row is needed, just clamp to the
        // provided range. Not generic but wil have to work for now.

        unsigned hz_secondline = min(hz_start+1, in_stft_size-1);

        float2 c;
        float val1, val2, val3, val4;
        if (ts_start == (unsigned)-1)
        {
            val1 = 0;
            val3 = 0;
        }
        else
        {
            c = inStft.elem(make_elemSize3_t( hz_start, ts_start, 0 ));
            val1 = sqrt(c.x*c.x + c.y*c.y);

            c = inStft.elem(make_elemSize3_t( hz_secondline, ts_start, 0 ));
            val3 = sqrt(c.x*c.x + c.y*c.y);
        }

        c = inStft.elem(make_elemSize3_t( hz_start, ts_start+1, 0 ));
        val2 = sqrt(c.x*c.x + c.y*c.y);

        c = inStft.elem(make_elemSize3_t( hz_secondline, ts_start+1, 0 ));
        val4 = sqrt(c.x*c.x + c.y*c.y);

        // Perform a kind of bicubic interpolation
        p = 3*p*p-2*p*p*p; // p and q are saturated, these equations compute
        q = 3*q*q-2*q*q*q; // an 'S' curve from 0 to 1.
        val = .07f*((val1*(1-q)+val2*q)*(1-p) + (val3*(1-q)+val4*q)*p);

        //ff = (hz_write-20)/22050.f;
        //const float f0 = 2.0f + 35*ff*ff*ff;
        val*=0.03f*hz_write;
    }

    val /= in_stft_size;

    elemSize3_t writePos = make_elemSize3_t( x, y, 0 );
    outBlock.e( writePos ) = val;
}


extern "C"
void expandCompleteStft( cudaPitchedPtrType<float2> inStft,
                 cudaPitchedPtrType<float> outBlock,
                 float out_min_hz,
                 float out_max_hz,
                 float out_stft_size,
                 float out_offset,
                 float in_min_hz,
                 float in_max_hz,
                 unsigned in_stft_size,
                 unsigned cuda_stream)
{
    dim3 block(32,1,1);
    dim3 grid( outBlock.getNumberOfElements().x/block.x, outBlock.getNumberOfElements().y, 1);

    if(grid.x>65535 || grid.y>65535 || 0!=(in_stft_size%32)) {
        printf("====================\n"
               "Invalid argument, expandCompleteStft.\n"
               "grid.x=%u || grid.y=%u || in_stft_size=%u\n"
               "===================\n",
               grid.x, grid.y, in_stft_size
               );
        return;
    }

    float start = out_min_hz;
    float steplogsize = log(out_max_hz)-log(out_min_hz);

    kernel_expand_complete_stft<<<grid, block, cuda_stream>>>(
        inStft, outBlock,
        start,
        steplogsize,
        out_stft_size,
        out_offset,
        in_min_hz,
        in_max_hz,
        in_stft_size );
}
