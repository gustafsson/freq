#ifndef DRAWNWAVEFORMDEF_H
#define DRAWNWAVEFORMDEF_H

#include "resample.h"
#include "drawnwaveformkernel.h"

/**
 Plot the waveform on the matrix.

 Not coalesced, could probably be optimized.
 */
template<typename Reader, typename Writer>
RESAMPLE_CALL void draw_waveform_elem(
        unsigned writePos_x,
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, unsigned readstop, float scaling )
{
    typedef typename Writer::Position WritePos;
    typedef typename Writer::Size WriteSize;
    WriteSize matrix_sz = out_waveform_matrix.numberOfElements();

    unsigned readPos1 = writePos_x * blob;
    unsigned readPos2 = (writePos_x + 1) * blob;
    typename Writer::Position writePos;

    if( writePos_x >= matrix_sz.width || readPos1 >= readstop )
        return;

    float blobinv = 1.f/blob;

    for (unsigned read_x = readPos1; read_x<readPos2 && read_x < readstop; ++read_x)
    {
        float v = in_waveform.read( read_x );
        v *= scaling;
#if defined(_WIN32) && !defined(USE_CUDA)
        v = max(-1.f, min(1.f, v));
#else
        v = fmaxf(-1.f, fminf(1.f, v));
#endif
        float y = (v+1.f)*.5f*(matrix_sz.height-1.f);
        unsigned y1 = (unsigned)y;
        unsigned y2 = y1+1;
        if (y2 >= matrix_sz.height)
        {
            y2 = matrix_sz.height - 1;
            y1 = y2 - 1;
        }
        float py = y-y1;


#ifdef __CUDACC__
        typedef float2 WriteType;
#define MakeWriteType make_float2
#else
        typedef typename Writer::T WriteType;
#define MakeWriteType WriteType
#endif


        writePos = WritePos( writePos_x, y1 );
        WriteType& w1 = (WriteType&)out_waveform_matrix.ref( writePos );
        w1 += MakeWriteType(0.8f*blobinv * (1.f-py), 0);

        writePos = WritePos( writePos_x, y2 );
        WriteType& w2 = (WriteType&)out_waveform_matrix.ref( writePos );
        w2 += MakeWriteType(0.8f*blobinv * py, 0);
    }
}


template<typename Reader, typename Writer>
RESAMPLE_CALL void draw_waveform_with_lines_elem(
        unsigned writePos_x,
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, unsigned readstop, float scaling )
{
    typedef typename Writer::Position WritePos;
    typedef typename Writer::Size WriteSize;
    WriteSize matrix_sz = out_waveform_matrix.numberOfElements();

    unsigned readPos = writePos_x * blob;
    float px = writePos_x * blob - readPos;

    if( writePos_x >= matrix_sz.width || readPos >= readstop )
        return;

    float blobinv = 1.f/blob;

    float v1 = in_waveform.read( readPos );
    float v2 = in_waveform.read( readPos+1 );
    float v = v1*(1-px) + v2*px;
    v *= scaling;
#if defined(_WIN32) && !defined(USE_CUDA)
        v = max(-1.f, min(1.f, v));
#else
        v = fmaxf(-1.f, fminf(1.f, v));
#endif
    float y = (v+1.f)*.5f*(matrix_sz.height-1.f);
    unsigned y1 = (unsigned)y;
    unsigned y2 = y1+1;
    if (y2 >= matrix_sz.height)
    {
        y2 = matrix_sz.height - 1;
        y1 = y2 - 1;
    }
    float py = y-y1;

#ifdef __CUDACC__
        typedef float2 WriteType;
#else
        typedef typename Writer::T WriteType;
#endif

    WritePos writePos( writePos_x, y1 );
    WriteType& w1 = (WriteType&)out_waveform_matrix.ref( writePos );
    w1 += MakeWriteType(0.8f*blobinv * (1.f-py), 0);

    writePos = WritePos( writePos_x, y2 );
    WriteType& w2 = (WriteType&)out_waveform_matrix.ref( writePos );
    w2 += MakeWriteType(0.8f*blobinv * py, 0);
}

#endif // DRAWNWAVEFORMDEF_H
