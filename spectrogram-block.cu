#include <stdio.h>
#include "spectrogram-block.cu.h"

__global__ void kernel_merge(
                cudaPitchedPtrType<float> inBlock,
                cudaPitchedPtrType<float> outBlock,
                float resample_width,
                float resample_height,
                float in_offset,
                float out_offset,
                float in_valid_samples)
{
    elemSize3_t writePos;
    if( !outBlock.unwrapCudaGrid( writePos ))
        return;

    float val = 0;
    //unsigned n = 0;

    if (writePos.x>=out_offset)
    {
        for (float x = 0; x < resample_width; x++)
        {
            float s = in_offset + x + resample_width*(writePos.x-out_offset);
            if ( s >= in_offset + in_valid_samples )
                continue;

            for (float y = 0; y < resample_height; y++)
            {
                float t = y + resample_height*writePos.y;

                elemSize3_t readPos = make_elemSize3_t( s, t, 0 );
                if ( inBlock.valid(readPos) ) {
                    val += inBlock.elem(readPos);

                    outBlock.e( writePos ) = val;
                    return;

                    //n ++;
                }
            }
        }
    }
/*
    if (0<n) {
        val/=n;
        outBlock.elem( writePos ) = val;
    }*/
}

extern "C"
void blockMerge( cudaPitchedPtrType<float> inBlock,
                 cudaPitchedPtrType<float> outBlock,
                 float in_sample_rate,
                 float out_sample_rate,
                 float in_frequency_resolution,
                 float out_frequency_resolution,
                 float in_offset,
                 float out_offset,
                 float in_valid_samples,
                 unsigned cuda_stream)
{
    dim3 grid, block;
    unsigned block_size = 128;

    outBlock.wrapCudaGrid2D( block_size, grid, block );

    float resample_width = in_sample_rate/out_sample_rate;
    float resample_height = in_frequency_resolution/out_frequency_resolution;

    kernel_merge<<<grid, block, cuda_stream>>>(
        inBlock, outBlock,
        resample_width,
        resample_height,
        in_offset, out_offset, in_valid_samples );
}

__global__ void kernel_merge_chunk(
                cudaPitchedPtrType<float2> inChunk,
                cudaPitchedPtrType<float> outBlock,
                float resample_width,
                float resample_height,
                float in_offset,
                float out_offset,
                unsigned n_valid_samples )
{
    elemSize3_t writePos;
    if( !outBlock.unwrapCudaGrid( writePos ))
        return;

    float val = 0;
    //unsigned n = 0;

    if (writePos.x>=out_offset)
    {
        for (float x = 0; x < resample_width; x++)
        {
            float s = in_offset + x + resample_width*(writePos.x-out_offset);
            if ( s >= in_offset + n_valid_samples )
                continue;

            for (float y = 0; y < resample_height; y++)
            {
                float t = y + resample_height*writePos.y;

                elemSize3_t readPos = make_elemSize3_t( s, t, 0 );
                if ( inChunk.valid(readPos) ) {
                    float2 c = inChunk.elem(readPos);
                    val += sqrt(c.x*c.x + c.y*c.y);

 outBlock.e( writePos ) = val;
 return;
/*
  TODO use command line argument "yscale"
                        case Yscale_Linear:
                            v[2][df] = amplitude;
                            break;
                        case Yscale_ExpLinear:
                            v[2][df] = amplitude * exp(.001*fi);
                            break;
                        case Yscale_LogLinear:
                            v[2][df] = amplitude;
                            v[2][df] = log(1+fabsf(v[2][df]))*(v[2][df]>0?1:-1);
                            break;
                        case Yscale_LogExpLinear:
                            v[2][df] = amplitude * exp(.001*fi);
                            v[2][df] = log(1+fabsf(v[2][df]))*(v[2][df]>0?1:-1);
                            */

  /*                  n ++;*/
                }
            }
        }
    }
/*
    __syncthreads();

    if (0<n) {
        val/=n;
        outBlock.e( writePos ) = val;
    }*/
}

/*
#define WARP 32

__global__ void kernel_merge_chunk(
                cudaPitchedPtrType<float2> inChunk,
                cudaPitchedPtrType<float> outBlock,
                float resample_width,
                float resample_height,
                float in_offset,
                float out_offset)
{
    elemSize3_t writePos;
    if( !outBlock.unwrapCudaGrid( writePos ))
        return;

    __shared__ float val[WARP] = 0;
    unsigned n = 0;

    if (writePos.x>=out_offset)
    {
        for (float x = 0; x < resample_width; x++)
        {
            for (float y = 0; y < resample_height; y++)
            {
                float s = in_offset + x + resample_width*(writePos.x-out_offset);
                float t = y + resample_height*writePos.y;

                elemSize3_t readPos = make_elemSize3_t( s, t, 0 );
                if ( inChunk.valid(readPos) ) {
                    unsigned o = inChunk.eOffs(readPos);
                    float* i = (float*)inChunk.ptr();
                    i[2*o + WARP%2];
                    float2 c
                    //val = max(val, sqrt(c.x*c.x + c.y*c.y)); n=0;
                    val += sqrt(c.x*c.x + c.y*c.y);
                    //val += c.x;

                    //outBlock.e( writePos ) = val;
                    //return;

                    n ++;
                }
            }
        }
    }

    __syncthreads();

    if (0<n && threadIdx.x < WARP) {
        val/=n;
        outBlock.e( writePos ) = val;
    }
}*/

extern "C"
void blockMergeChunk( cudaPitchedPtrType<float2> inChunk,
                 cudaPitchedPtrType<float> outBlock,
                 float in_sample_rate,
                 float out_sample_rate,
                 float in_frequency_resolution,
                 float out_frequency_resolution,
                 float in_offset,
                 float out_offset,
                 unsigned n_valid_samples,
                 unsigned cuda_stream)
{
    dim3 grid, block;
    unsigned block_size = 128;

    outBlock.wrapCudaGrid2D( block_size, grid, block );

    float resample_width = in_sample_rate/out_sample_rate;
    float resample_height = in_frequency_resolution/out_frequency_resolution;

    if(0) {
        elemSize3_t sz_o = outBlock.getNumberOfElements();
        elemSize3_t sz_i = inChunk.getNumberOfElements();
        fprintf(stdout,"sz_o (%d, %d, %d)\tsz_i (%d, %d, %d)\n", sz_o.x, sz_o.y, sz_o.z, sz_i.x, sz_i.y, sz_i.z );


        fprintf(stdout,"grid (%d, %d, %d)\tblock (%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z );
        fprintf(stdout,"in sr %g, out sr %g, in f %g, out f %g, in o %g, out o %g\n\tw=%g, h=%g\n",
            in_sample_rate, out_sample_rate,
            in_frequency_resolution, out_frequency_resolution,
            in_offset, out_offset,
            resample_width, resample_height);
        fprintf(stdout,"outBlock(%d,%d,%d) pitch %lu\n",
            outBlock.getNumberOfElements().x,
            outBlock.getNumberOfElements().y,
            outBlock.getNumberOfElements().z,
            outBlock.getCudaPitchedPtr().pitch );
        fprintf(stdout,"inChunk(%d,%d,%d) pitch %lu\n",
            inChunk.getNumberOfElements().x,
            inChunk.getNumberOfElements().y,
            inChunk.getNumberOfElements().z,
            inChunk.getCudaPitchedPtr().pitch );
        fflush(stdout);

    }

    kernel_merge_chunk<<<grid, block, cuda_stream>>>(
        inChunk, outBlock,
        resample_width,
        resample_height,
        in_offset, out_offset, n_valid_samples );
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
            y = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    unsigned nFrequencies = outBlock.getNumberOfElementsD().y;
    if( y >= nFrequencies )
        return;

    float ff = y/(float)nFrequencies;
    float hz_out = start*exp(ff*steplogsize);

    float max_stft_hz = inStft.getNumberOfElementsD().x/2;
    float read_f = hz_out/max_stft_hz;

    float2 c;

    float p = read_f*inStft.getNumberOfElementsD().x;
    elemSize3_t readPos = make_elemSize3_t( p, 0, 0 );
    inStft.clamp(readPos);
    c = inStft.elem(readPos);
    float val1 = sqrt(c.x*c.x + c.y*c.y);

    readPos.x++;
    inStft.clamp(readPos);
    c = inStft.elem(readPos);
    float val2 = sqrt(c.x*c.x + c.y*c.y);

    p-=(unsigned)p;
    float val = val1*(1-p)+val2*p;

    elemSize3_t writePos = make_elemSize3_t( 0, y, 0 );
    for (writePos.x=out_offset; writePos.x<out_offset + out_length && writePos.x<outBlock.getNumberOfElementsD().x;writePos.x++)
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
    dim3 grid( INTDIV_CEIL(outBlock.getNumberOfElements().x, block.y), 1, 1);

    if(grid.x>65535) {
        printf("====================\nInvalid argument, number of floats in complex signal must be less than 65535*256.\n===================\n");
        return;
    }

    float start = min_hz*outBlock.getNumberOfElements().x;
    float steplogsize = log(max_hz)-log(min_hz);

    kernel_expand_stft<<<grid, block, cuda_stream>>>(
        inStft, outBlock,
        start,
        steplogsize,
        out_offset,
        out_length );
}
