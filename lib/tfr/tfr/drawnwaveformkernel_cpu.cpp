#ifndef USE_CUDA

#include "resamplecpu.h"
#include "drawnwaveformkerneldef.h"
#include <stdio.h>
#include "msc_stdc.h"


void drawWaveform(
        DataStorage<float>::ptr in_waveformp,
        Tfr::ChunkData::ptr out_waveform_matrixp,
        float blob, int readstop, float maxValue, float writeposoffs )
{
    CpuMemoryReadOnly<float, 1> in_waveform = CpuMemoryStorage::ReadOnly<1>( in_waveformp );
    CpuMemoryReadWrite<Tfr::ChunkElement, 2> out_waveform_matrix = CpuMemoryStorage::ReadWrite<2>( out_waveform_matrixp );

    int w = out_waveform_matrixp->size().width;
    if (blob > 1)
    {
        for(int writePos_x=0; writePos_x<w; ++writePos_x)
            draw_waveform_elem( writePos_x, in_waveform, out_waveform_matrix, blob, readstop, 1.f/maxValue, writeposoffs );
    }
    else
    {
        for(int writePos_x=0; writePos_x<w; ++writePos_x)
            draw_waveform_with_lines_elem( writePos_x, in_waveform, out_waveform_matrix, blob, readstop, 1.f/maxValue, writeposoffs );
    }
}


void drawWaveform(
        DataStorage<float>::ptr in_waveformp,
        DataStorage<float>::ptr out_waveform_matrixp,
        float blob, int readstop, float maxValue, float writeposoffs, float y0 )
{
    CpuMemoryReadOnly<float, 1> in_waveform = CpuMemoryStorage::ReadOnly<1>( in_waveformp );
    CpuMemoryReadWrite<float, 2> out_waveform_matrix = CpuMemoryStorage::ReadWrite<2>( out_waveform_matrixp );

    int w = out_waveform_matrixp->size().width;
    if (blob > 1)
    {
        for(int writePos_x=0; writePos_x<w; ++writePos_x)
            draw_waveform_elem( writePos_x, in_waveform, out_waveform_matrix, blob, readstop, 1.f/maxValue, writeposoffs, y0 );
    }
    else
    {
        for(int writePos_x=0; writePos_x<w; ++writePos_x)
            draw_waveform_with_lines_elem( writePos_x, in_waveform, out_waveform_matrix, blob, readstop, 1.f/maxValue, writeposoffs, y0 );
    }
}

#endif // USE_CUDA
