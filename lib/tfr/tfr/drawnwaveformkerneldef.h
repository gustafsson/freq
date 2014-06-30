#ifndef DRAWNWAVEFORMDEF_H
#define DRAWNWAVEFORMDEF_H

#include "resample.h"
#include "drawnwaveformkernel.h"

#if defined(_WIN32) && !defined(USE_CUDA)
#include <math.h>
#endif


template<typename T>
T e(float v);

template <>
float e<float>(float v) { return v; }

#ifdef __CUDACC__
template <>
float2 e<float2>(float v) { return make_float2(v,0); }
#else
template <>
std::complex<float> e<std::complex<float> >(float v) { return std::complex<float>(v,0); }
#endif


/**
 Plot the waveform on the matrix.

 Not coalesced, could probably be optimized.
 */
template<typename Reader, typename Writer>
RESAMPLE_CALL void draw_waveform_pts_elem(
        int writePos_x,
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, int readstop, float scaling )
{
    typedef typename Writer::Position WritePos;
    typedef typename Writer::Size WriteSize;
    WriteSize matrix_sz = out_waveform_matrix.numberOfElements();

    int readPos1 = writePos_x * blob;
    int readPos2 = (writePos_x + 1) * blob;
    typename Writer::Position writePos;

    if( writePos_x >= matrix_sz.width || readPos1 >= readstop )
        return;

    float blobinv = 1.f/blob;

    for (int read_x = readPos1; read_x<readPos2 && read_x < readstop; ++read_x)
    {
        float v = in_waveform.read( read_x );
        v *= scaling;
        v = fmax(-1.f, fmin(1.f, v));

        float y = (v+1.f)*.5f*(matrix_sz.height-1.f);
        int y1 = (int)y;
        int y2 = y1+1;
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
        w1 += e<WriteType>(0.8f*blobinv * (1.f-py));

        writePos = WritePos( writePos_x, y2 );
        WriteType& w2 = (WriteType&)out_waveform_matrix.ref( writePos );
        w2 += e<WriteType>(0.8f*blobinv * py);
    }
}


/**
 Plot the waveform on the matrix.

 Not coalesced, could probably be optimized.
 */
template<typename Reader, typename Writer>
RESAMPLE_CALL void draw_waveform_elem(
        int writePos_x,
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, int readstop, float scaling, float writeposoffs, float y0=0 )
{
    float A1 = 100.f; // 0.2f;
    float A2 = 10.f; // 0.01f

    typedef typename Writer::Position WritePos;
    typedef typename Writer::Size WriteSize;

    WriteSize matrix_sz = out_waveform_matrix.numberOfElements();
    if (writePos_x >= matrix_sz.width)
        return;

    int readPos1 = (writePos_x + writeposoffs) * blob;
    int readPos2 = (writePos_x + writeposoffs + 1) * blob;
    int readCenter = (readPos1+readPos2)/2;

    if (readCenter < 0)
        return;
    if (readCenter >= readstop)
        return;

    if (readPos1 < 0) readPos1 = 0;
    if (readPos2 >= readstop) readPos2 = readstop-1;

    float maxy = 0;
    float miny = matrix_sz.height;
    float blobinv = 1.f/blob;

    for (int i=0; i<matrix_sz.height; i++) {
        out_waveform_matrix.ref( WritePos( writePos_x, i ) ) = e<typename Writer::T>(0);
    }

    for (int read_x = readPos1; read_x <= readPos2 && read_x < readstop; ++read_x)
    {
        float v = in_waveform.read( read_x );

        v -= y0;
        v *= scaling;

        float y = v*(matrix_sz.height-1.f);
        if (y>maxy) maxy = y;
        if (y<miny) miny = y;

        if (0.f <= y && y < matrix_sz.height)
        {
            int y1 = (int)y;
            int y2 = y1+1;

            if (y2 >= matrix_sz.height)
                y2 = matrix_sz.height - 1;

            float py = y-y1;


    #ifdef __CUDACC__
            typedef float2 WriteType;
    #else
            typedef typename Writer::T WriteType;
    #endif

            WriteType& w1 = (WriteType&)out_waveform_matrix.ref( WritePos( writePos_x, y1 ) );
            w1 += e<WriteType>(A1*blobinv * (1.f-py));

            WriteType& w2 = (WriteType&)out_waveform_matrix.ref( WritePos( writePos_x, y2 ) );
            w2 += e<WriteType>(A1*blobinv * py);
        }
    }

    if (0.f <= miny && maxy <= matrix_sz.height)
    {
        int y1 = floor(miny);
        int y2 = ceil(maxy);

        if (y2 == matrix_sz.height)
            y2 = matrix_sz.height - 1;

        if (y2-y1 < 1)
        {
            if (y2>=5)
                y1 = y2-5;
            else
                y2 = y1+5;
        }

        for (int y=y1; y<=y2; ++y)
            out_waveform_matrix.ref( WritePos( writePos_x, y ) ) += e<typename Writer::T>(A2);
    }
}


template<typename Reader, typename Writer>
RESAMPLE_CALL void draw_waveform_with_lines_elem(
        int writePos_xu,
        Reader in_waveform,
        Writer out_waveform_matrix, float blob, int readstop, float scaling, float writeposoffs, float y0=0 )
{
    float A = 10.f; // 0.02f

    typedef typename Writer::Position WritePos;
    typedef typename Writer::Size WriteSize;
    WriteSize matrix_sz = out_waveform_matrix.numberOfElements();
    if( writePos_xu >= matrix_sz.width )
        return;

    float readPos_x = writePos_xu + writeposoffs;
    int readPos = readPos_x * blob;
    float px1_0 = (readPos_x-1) * blob - readPos;
    float px2_0 = (readPos_x+1) * blob - readPos;
    float px1 = px1_0;
    float px2 = px2_0;
    if (px2>1.f) { px2 = 1.f; }
    if (px1<0.f) { px1 = 0.f; }

    if (readPos < 0 || readPos >= readstop )
        return;

    float v1 = in_waveform.read( readPos );
    float v2 = in_waveform.read( readPos+1 );
    float w1 = v1*(1-px1) + v2*px1;
    float w2 = v1*(1-px2) + v2*px2;
    w1 -= y0;
    w2 -= y0;
    w1 *= scaling;
    w2 *= scaling;
    float fy1 = w1*(matrix_sz.height-1.f);
    float fy2 = w2*(matrix_sz.height-1.f);

    float y_per_x = (fy2-fy1)/(px2-px1);
    float dy = (px2_0-px1_0)*y_per_x;
    float my = fy1 - (px1-px1_0)*y_per_x + dy/2;

    dy = fabsf(dy);
    //float dy = fy2-fy1;
    //float my = 0.5f*(fy2+fy1);

    if (dy < 6)
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
    }

    int y1 = (int)fy1;
    int y2 = (int)ceil(fy2);

    float invdy = 2.f/dy;
    for (int y=y1; y<y2; ++y)
    {
        float py = y;
        py = fmax(0.f, 1.f - fabsf(my - y)*invdy);
        out_waveform_matrix.ref( WritePos( writePos_xu, y ) ) = e<typename Writer::T>(A*py);
    }
}

#endif // DRAWNWAVEFORMDEF_H
