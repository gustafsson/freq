#include <stdio.h>
#include "heightmap/block.cu.h"

#include <resample.cu.h>

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif


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
    __device__ float operator()( float2 v, uint2 const& /*dataPosition*/ )
    {
        // slightly faster than sqrtf(f) unless '--use_fast_math' is specified
        // to nvcc
        // return f*rsqrtf(f);
        //return log2f(0.01f+sqrtf(v.x*v.x + v.y*v.y)) - log2f(0.01f);
        return 0.4f*powf(v.x*v.x + v.y*v.y, 0.1);
        //return (sqrtf(v.x*v.x + v.y*v.y));
    }
};

#define M_PIf ((float)M_PI)

class WeightFetcher
{
public:
    template<typename Reader>
    __device__ float operator()( float2 const& p, Reader& reader )
    {
        float v = InterpolateFetcher<float, ConverterLogAmplitude>()( p, reader );
        float phase1 = InterpolateFetcher<float, ConverterPhase>()( p, reader );
        float phase2 = InterpolateFetcher<float, ConverterPhase>()( make_float2(p.x, p.y+1), reader );
        float phasediff = phase2 - phase1;
        if (phasediff < -M_PIf ) phasediff += 2*M_PIf;
        if (phasediff > M_PIf ) phasediff -= 2*M_PIf;

        float k = exp2f(-fabsf(phasediff));

        return v * k;
    }
};

void blockResampleChunk( cudaPitchedPtrType<float2> input,
                 cudaPitchedPtrType<float> output,
                 uint2 validInputs,
                 float4 inputRegion,
                 float4 outputRegion,
                 Heightmap::ComplexInfo transformMethod
                 )
{
    elemSize3_t sz_input = input.getNumberOfElements();
    elemSize3_t sz_output = output.getNumberOfElements();

    uint4 validInputs4 = make_uint4( validInputs.x, 0, validInputs.y, sz_input.y );
    uint2 validOutputs = make_uint2( sz_output.x, sz_output.y );

    switch (transformMethod)
    {
    case Heightmap::ComplexInfo_Amplitude_Weighted:
    {
        resample2d_fetcher<float, float2, float, WeightFetcher, AssignOperator<float> >(
                input,
                output,
                validInputs4,
                validOutputs,
                inputRegion,
                outputRegion
        );
        break;
    }
    case Heightmap::ComplexInfo_Amplitude_Non_Weighted:
        resample2d<float2, float, ConverterLogAmplitude, AssignOperator<float> >(
                input,
                output,
                validInputs4,
                validOutputs,
                inputRegion,
                outputRegion
        );
        break;
    case Heightmap::ComplexInfo_Phase:
        resample2d<float2, float, ConverterPhase, AssignOperator<float> >(
                    input,
                    output,
                    validInputs4,
                    validOutputs,
                    inputRegion,
                    outputRegion
            );
    }
}


class StftFetcher
{
public:
    template<typename Reader>
    __device__ float operator()( float2 const& p, Reader& reader )
    {
        float2 q;
        // exp2f (called in 'getFrequency') is only 4 multiplies for arch 1.x
        // so these are fairly cheap operations. One reciprocal called in
        // 'getFrequencyScalar' is just as fast.
        float hz = outputAxis.getFrequency( p.x );
        q.x = inputAxis.getFrequencyScalar( hz );
        q.y = p.y;
        float r = InterpolateFetcher<float, ConverterLogAmplitude>()( q, reader );
        return r*factor;
    }

    Tfr::FreqAxis inputAxis;
    Tfr::FreqAxis outputAxis;
    float factor;
};


void resampleStft( cudaPitchedPtrType<float2> input,
                   cudaPitchedPtrType<float> output,
                   float4 inputRegion,
                   float4 outputRegion,
                   Tfr::FreqAxis inputAxis,
                   Tfr::FreqAxis outputAxis )
{
    elemSize3_t sz_input = input.getNumberOfElements();
    elemSize3_t sz_output = output.getNumberOfElements();

    // We wan't to do our own translation from coordinate position to matrix
    // position in the input matrix. By giving bottom=0 and top=2 we tell
    // 'resample2d_fetcher' to only translate to a normalized reading position
    // [0, height-1) along the input x-axis. 'resample2d_fetcher' does to
    // transpose for us.
    uint4 validInputs = make_uint4( 0, 0, 2, sz_input.y );
    uint2 validOutputs = make_uint2( sz_output.x, sz_output.y );

    StftFetcher fetcher;
    fetcher.inputAxis = inputAxis;
    fetcher.outputAxis = outputAxis;
    fetcher.factor = 0.22; // makes it roughly equal height to Cwt

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


extern "C"
void blockMerge( cudaPitchedPtrType<float> inBlock,
                 cudaPitchedPtrType<float> outBlock,
                 float4 in_area,
                 float4 out_area)
{
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
