#include "drawnwaveform.cu.h"
#include <stdio.h>

#include "cudaglobalstorage.h"

/**
 Plot the waveform on the matrix.

 Not coalesced, could probably be optimized.
 */
__global__ void kernel_draw_waveform(
        cudaPitchedPtrType<float> in_waveform,
        cudaPitchedPtrType<float2> out_waveform_matrix,
        float blob, unsigned readstop, float scaling );


__global__ void kernel_draw_waveform_with_lines(
        cudaPitchedPtrType<float> in_waveform,
        cudaPitchedPtrType<float2> out_waveform_matrix,
        float blob, unsigned readstop, float scaling );


void drawWaveform(
        DataStorage<float>::Ptr in_waveformp,
        Tfr::ChunkData::Ptr out_waveform_matrixp,
        float blob, unsigned readstop, float maxValue )
{
    cudaPitchedPtrType<float> in_waveform(CudaGlobalStorage::ReadOnly<1>( in_waveformp ).getCudaPitchedPtr());

    cudaPitchedPtrType<float2> out_waveform_matrix(CudaGlobalStorage::ReadWrite<2>( in_waveformp ).getCudaPitchedPtr());

    cudaMemset( out_waveform_matrix.ptr(), 0, out_waveform_matrix.getTotalBytes() );

    unsigned w = out_waveform_matrix.getNumberOfElements().x;
    dim3 block(drawWaveform_BLOCK_SIZE, 1, 1);
    dim3 grid(int_div_ceil(w, block.x), 1, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    if (blob > 1)
    {
        printf("blob > 1: %g", blob);
        kernel_draw_waveform<<<grid, block, 0, 0>>>( in_waveform, out_waveform_matrix, blob, readstop, 1.f/maxValue );
    }
    else
    {
        printf("blob <= 1: %g", blob);
        kernel_draw_waveform_with_lines<<<grid, block, 0, 0>>>( in_waveform, out_waveform_matrix, blob, readstop, 1.f/maxValue );
    }
}


__global__ void kernel_draw_waveform(
        cudaPitchedPtrType<float> in_waveform,
        cudaPitchedPtrType<float2> out_waveform_matrix, float blob, unsigned readstop, float scaling )
{
    elemSize_t writePos_x = blockIdx.x * blockDim.x + threadIdx.x;
    elemSize3_t matrix_sz = out_waveform_matrix.getNumberOfElements();
    elemSize_t readPos1 = writePos_x * blob;
    elemSize_t readPos2 = (writePos_x + 1) * blob;
    elemSize3_t writePos;

    if( writePos_x >= matrix_sz.x || readPos1 >= readstop )
        return;

    float blobinv = 1.f/blob;

    for (elemSize_t read_x = readPos1; read_x<readPos2 && read_x < readstop; ++read_x)
    {
        elemSize3_t readPos = make_elemSize3_t(read_x, 0, 0);

        float v = in_waveform.elem( readPos );
        v *= scaling;
        v = fmaxf(-1.f, fminf(1.f, v));
        float y = (v+1.f)*.5f*(matrix_sz.y-1.f);
        elemSize_t y1 = (elemSize_t)y;
        elemSize_t y2 = y1+1;
        if (y2 >= matrix_sz.y)
        {
            y2 = matrix_sz.y - 1;
            y1 = y2 - 1;
        }
        float py = y-y1;

        writePos = make_elemSize3_t( writePos_x, y1, 0 );
        out_waveform_matrix.e( writePos ).x += 0.8f*blobinv * (1.f-py);

        writePos = make_elemSize3_t( writePos_x, y2, 0 );
        out_waveform_matrix.e( writePos ).x += 0.8f*blobinv * py;
    }
}


__global__ void kernel_draw_waveform_with_lines(
        cudaPitchedPtrType<float> in_waveform,
        cudaPitchedPtrType<float2> out_waveform_matrix, float blob, unsigned readstop, float scaling )
{
    elemSize_t writePos_x = blockIdx.x * blockDim.x + threadIdx.x;
    elemSize3_t matrix_sz = out_waveform_matrix.getNumberOfElements();
    elemSize_t readPos = writePos_x * blob;
    float px = writePos_x * blob - readPos;

    if( writePos_x >= matrix_sz.x || readPos >= readstop )
        return;

    float blobinv = 1.f/blob;

    float v1 = in_waveform.elem( make_elemSize3_t(readPos, 0, 0) );
    float v2 = in_waveform.elem( make_elemSize3_t(readPos+1, 0, 0) );
    float v = v1*(1-px) + v2*px;
    v *= scaling;
    v = fmaxf(-1.f, fminf(1.f, v));
    float y = (v+1.f)*.5f*(matrix_sz.y-1.f);
    elemSize_t y1 = (elemSize_t)y;
    elemSize_t y2 = y1+1;
    if (y2 >= matrix_sz.y)
    {
        y2 = matrix_sz.y - 1;
        y1 = y2 - 1;
    }
    float py = y-y1;

    elemSize3_t writePos = make_elemSize3_t( writePos_x, y1, 0 );
    out_waveform_matrix.e( writePos ).x += 0.8f*blobinv * (1.f-py);

    writePos = make_elemSize3_t( writePos_x, y2, 0 );
    out_waveform_matrix.e( writePos ).x += 0.8f*blobinv * py;
}
