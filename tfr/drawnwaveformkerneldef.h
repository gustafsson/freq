#ifndef DRAWNWAVEFORMDEF_H
#define DRAWNWAVEFORMDEF_H

#include "resample.h"
#include "drawnwaveformkernel.h"

#if defined(_WIN32) && !defined(USE_CUDA)
#include <math.h>
#endif

#ifdef __CUDACC__
#define MakeWriteType make_float2
#else
#define MakeWriteType typename Writer::T
#endif

/**
 Plot the waveform on the matrix.

 Not coalesced, could probably be optimized.
 */
template<typename Reader, typename Writer>
RESAMPLE_CALL void draw_waveform_pts_elem(
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
        v = fmaxf(-1.f, fminf(1.f, v));

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


        writePos = WritePos( writePos_x, y1 );
        WriteType& w1 = (WriteType&)out_waveform_matrix.ref( writePos );
        w1 += MakeWriteType(0.8f*blobinv * (1.f-py), 0);

        writePos = WritePos( writePos_x, y2 );
        WriteType& w2 = (WriteType&)out_waveform_matrix.ref( writePos );
        w2 += MakeWriteType(0.8f*blobinv * py, 0);
    }
}

/**
 Plot the waveform on the matrix.

 Not coalesced, could probably be optimized.
 */
template<typename Reader, typename Writer>
RESAMPLE_CALL void draw_waveform_elem(
        unsigned writePos_xu,
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, unsigned readstop, float scaling, float writeposoffs )
{
    typedef typename Writer::Position WritePos;
    typedef typename Writer::Size WriteSize;
    WriteSize matrix_sz = out_waveform_matrix.numberOfElements();

    float writePos_x = writePos_xu + writeposoffs;
    unsigned readPos1 = writePos_x * blob;
    unsigned readPos2 = (writePos_x + 1) * blob;

    if( writePos_x >= matrix_sz.width || readPos1 >= readstop )
        return;

    float maxy = 0;
    float miny = matrix_sz.height;
    float blobinv = 1.f/blob;

    for (unsigned read_x = readPos1; read_x <= readPos2 && read_x < readstop; ++read_x)
    {
        float v = in_waveform.read( read_x );

        v *= scaling;
        v = fmaxf(-1.f, fminf(1.f, v));

        float y = (v+1.f)*.5f*(matrix_sz.height-1.f);
        if (y>maxy) maxy = y;
        if (y<miny) miny = y;

        if (0.f <= y && y <= matrix_sz.height)
        {
            unsigned y1 = (unsigned)y;
            unsigned y2 = y1+1;

            if (y2 >= matrix_sz.height)
                y2 = matrix_sz.height - 1;

            float py = y-y1;


    #ifdef __CUDACC__
            typedef float2 WriteType;
    #else
            typedef typename Writer::T WriteType;
    #endif

            WriteType& w1 = (WriteType&)out_waveform_matrix.ref( WritePos( writePos_x, y1 ) );
            w1 += MakeWriteType(0.2f*blobinv * (1.f-py), 0);

            WriteType& w2 = (WriteType&)out_waveform_matrix.ref( WritePos( writePos_x, y2 ) );
            w2 += MakeWriteType(0.2f*blobinv * py, 0);
        }
    }

    if (0.f <= miny && maxy <= matrix_sz.height)
    {
        unsigned y1 = floor(miny);
        unsigned y2 = ceil(maxy);

        if (y2 == matrix_sz.height)
            y2 = matrix_sz.height - 1;

        if (y2-y1 < 1)
        {
            if (y2>=5)
                y1 = y2-5;
            else
                y2 = y1+5;
        }

        for (unsigned y=y1; y<=y2; ++y)
            out_waveform_matrix.ref( WritePos( writePos_x, y ) ) += MakeWriteType(0.01f*blobinv, 0);
    }
}


template<typename Reader, typename Writer>
RESAMPLE_CALL void draw_waveform_with_lines_elem(
        unsigned writePos_xu,
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, unsigned readstop, float scaling, float writeposoffs )
{
    typedef typename Writer::Position WritePos;
    typedef typename Writer::Size WriteSize;
    WriteSize matrix_sz = out_waveform_matrix.numberOfElements();

    float writePos_x = writePos_xu + writeposoffs;
    unsigned readPos = writePos_x * blob;
    float px1_0 = (writePos_x-1) * blob - readPos;
    float px2_0 = (writePos_x+1) * blob - readPos;
    float px1 = px1_0;
    float px2 = px2_0;
    if (px2>1.f) { px2 = 1.f; }
    if (px1<0.f) { px1 = 0.f; }

    if( writePos_x >= matrix_sz.width || readPos >= readstop )
        return;

    float v1 = in_waveform.read( readPos );
    float v2 = in_waveform.read( readPos+1 );
    float w1 = v1*(1-px1) + v2*px1;
    float w2 = v1*(1-px2) + v2*px2;
    w1 *= scaling;
    w2 *= scaling;
    w1 = fmaxf(-1.f, fminf(1.f, w1));
    w2 = fmaxf(-1.f, fminf(1.f, w2));
    float fy1 = (w1+1.f)*.5f*(matrix_sz.height-1.f);
    float fy2 = (w2+1.f)*.5f*(matrix_sz.height-1.f);

    float y_per_x = (fy2-fy1)/(px2-px1);
    float dy = (px2_0-px1_0)*y_per_x;
    float my = fy1 - (px1-px1_0)*y_per_x + dy/2;

    //float dy = fy2-fy1;
    //float my = 0.5f*(fy2+fy1);

    if (fabsf(dy) < 6)
    {
        fy1 = my-3;
        fy2 = my+3;
        dy = 6;
    }

    if (fy2 >= matrix_sz.height - 1)
        fy2 = matrix_sz.height - 1;
    if (fy1 >= matrix_sz.height - 1)
        fy1 = matrix_sz.height - 1;
    if (fy2 < 0 )
        fy2 = 0;
    if (fy1 < 0 )
        fy1 = 0;

    if (fy1>fy2)
    {
        // swap
        fy2 += fy1;
        fy1 = fy2 - fy1;
        fy2 = fy2 - fy1;

        dy *= -1;
    }

    unsigned y1 = (unsigned)fy1;
    unsigned y2 = (unsigned)ceil(fy2);

    float invdy = 2.f/dy;
    for (unsigned y=y1; y<=y2; ++y)
    {
        float py = y;
        py = fmaxf(0.f, 1.f - fabsf(my - y)*invdy);
        out_waveform_matrix.ref( WritePos( writePos_x, y ) ) = MakeWriteType(0.02f*py, 0);
    }
}

#endif // DRAWNWAVEFORMDEF_H
