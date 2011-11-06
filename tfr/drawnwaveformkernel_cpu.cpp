#ifndef USE_CUDA

#include "resamplecpu.h"
#include "drawnwaveformkerneldef.h"
#include <stdio.h>


void drawWaveform(
        DataStorage<float>::Ptr in_waveformp,
        Tfr::ChunkData::Ptr out_waveform_matrixp,
        float blob, unsigned readstop, float maxValue )
{
    CpuMemoryReadOnly<float, 1> in_waveform = CpuMemoryStorage::ReadOnly<1>( in_waveformp );
    CpuMemoryReadWrite<Tfr::ChunkElement, 2> out_waveform_matrix = CpuMemoryStorage::ReadWrite<2>( out_waveform_matrixp );

    unsigned w = out_waveform_matrixp->size().width;
    if (blob > 1)
    {
        printf("blob > 1: %g", blob);
        for(unsigned writePos_x=0; writePos_x<w; ++writePos_x)
            draw_waveform_elem( writePos_x, in_waveform, out_waveform_matrix, blob, readstop, 1.f/maxValue );
    }
    else
    {
        printf("blob <= 1: %g", blob);
        for(unsigned writePos_x=0; writePos_x<w; ++writePos_x)
            draw_waveform_with_lines_elem( writePos_x, in_waveform, out_waveform_matrix, blob, readstop, 1.f/maxValue );
    }
}
#endif // USE_CUDA
