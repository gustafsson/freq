#include <stdio.h>
#include "spectrogram-block.cu.h"

__global__ void kernel_merge(
                cudaPitchedPtrType<float> outBlock,
                cudaPitchedPtrType<float2> inChunk,
                float resample_width,
                float resample_height,
                float in_offset,
                float out_offset)
{
    ushort3 threadPos;
    if( !outBlock.unwrapCudaGrid( threadPos ))
        return;

    float val = 0;
    unsigned n = 0;

    if (threadPos.x-out_offset >= 0)
    {
        for (float x = 0; x < resample_width; x++)
        {
            for (float y = 0; y < resample_height; y++)
            {
                float s = in_offset + x + resample_width*(threadPos.x-out_offset);
                float t = y + resample_height*threadPos.y;

                ushort3 readPos = make_ushort3( s, t, 0 );
                if ( inChunk.valid(readPos) ) {
                    float2 c = inChunk.e(readPos);
                    //val = max(val, sqrt(c.x*c.x + c.y*c.y)); n=0;
                    val += sqrt(c.x*c.x + c.y*c.y);
                    //val += c.x;

                    n ++;
                }
            }
        }
    }

    if (0<n) {
        val/=n;

        outBlock.e( threadPos ) = val;
    }
}


extern "C"
void blockMerge( cudaPitchedPtrType<float> outBlock,
                 cudaPitchedPtrType<float2> inChunk,
                 float in_sample_rate,
                 float out_sample_rate,
                 float in_frequency_resolution,
                 float out_frequency_resolution,
                 float in_offset,
                 float out_offset)
{
    dim3 grid, block;
    unsigned block_size = 128;
    ushort3 sz_o = outBlock.getNumberOfElements();
    ushort3 sz_i = inChunk.getNumberOfElements();
    fprintf(stdout,"sz_o (%d, %d, %d)\tsz_i (%d, %d, %d)\n", sz_o.x, sz_o.y, sz_o.z, sz_i.x, sz_i.y, sz_i.z );

    outBlock.wrapCudaGrid2D( block_size, grid, block );

    float resample_width = in_sample_rate/out_sample_rate;
    float resample_height = in_frequency_resolution/out_frequency_resolution;

    fprintf(stdout,"grid (%d, %d, %d)\tblock (%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z );
    fprintf(stdout,"in sr %g, out sr %g, in f %g, out f %g, in o %g, out o %g\n\tw=%g, h=%g\n",
        in_sample_rate, out_sample_rate,
        in_frequency_resolution, out_frequency_resolution,
        in_offset, out_offset,
        resample_width, resample_height);
    fflush(stdout);


    kernel_merge<<<grid, block>>>(
        outBlock, inChunk,
        resample_width,
        resample_height,
        in_offset, out_offset );
}
