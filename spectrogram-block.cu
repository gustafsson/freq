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
    elemSize3_t writePos;
    if( !outBlock.unwrapCudaGrid( writePos ))
        return;

    float val = 0;
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
                    float2 c = inChunk.elem(readPos);
                    //val = max(val, sqrt(c.x*c.x + c.y*c.y)); n=0;
                    val += sqrt(c.x*c.x + c.y*c.y);
                    //val += c.x;
                    outBlock.e( writePos ) = val;
                    return;

                    n ++;
                }
            }
        }
    }

    if (0<n) {
        val/=n;
        outBlock.e( writePos ) = val;
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
    elemSize3_t sz_o = outBlock.getNumberOfElements();
    elemSize3_t sz_i = inChunk.getNumberOfElements();
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


    kernel_merge<<<grid, block>>>(
        outBlock, inChunk,
        resample_width,
        resample_height,
        in_offset, out_offset );
}
