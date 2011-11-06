#include "resamplecuda.cu.h"
#include "cuda_vector_types_op.h"
#include "drawnwaveformkerneldef.h"
#include <stdio.h>


template<typename Reader, typename Writer>
__global__ void kernel_draw_waveform(
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, unsigned readstop, float scaling );


template<typename Reader, typename Writer>
__global__ void kernel_draw_waveform_with_lines(
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, unsigned readstop, float scaling );


void drawWaveform(
        DataStorage<float>::Ptr in_waveformp,
        Tfr::ChunkData::Ptr out_waveform_matrixp,
        float blob, unsigned readstop, float maxValue )
{
    CudaGlobalReadOnly<float, 1> in_waveform = CudaGlobalStorage::ReadOnly<1>( in_waveformp );
    CudaGlobalReadWrite<float2, 2> out_waveform_matrix( CudaGlobalStorage::ReadWrite<2>( out_waveform_matrixp ).getCudaPitchedPtr() );

    unsigned w = out_waveform_matrixp->size().width;
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


template<typename Reader, typename Writer>
__global__ void kernel_draw_waveform(
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, unsigned readstop, float scaling )
{
    unsigned writePos_x = blockIdx.x * blockDim.x + threadIdx.x;

    draw_waveform_elem(
            writePos_x,
            in_waveform,
            out_waveform_matrix, blob, readstop, scaling );
}


template<typename Reader, typename Writer>
__global__ void kernel_draw_waveform_with_lines(
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, unsigned readstop, float scaling )
{
    unsigned writePos_x = blockIdx.x * blockDim.x + threadIdx.x;

    draw_waveform_with_lines_elem(
            writePos_x,
            in_waveform,
            out_waveform_matrix, blob, readstop, scaling );
}
