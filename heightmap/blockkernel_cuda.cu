#include <resamplecuda.cu.h>

#include <complex>

template<>
RESAMPLE_CALL float ConverterAmplitude::
operator()( std::complex<float> w, DataPos const& /*dataPosition*/ )
{
    // slightly faster than sqrtf(f) unless '--use_fast_math' is specified
    // to nvcc
    // return f*rsqrtf(f);
    float2&v=(float2&)w;
    return sqrtf(v.x*v.x + v.y*v.y);
}


#include "blockkerneldef.h"

#include <stdio.h>


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
