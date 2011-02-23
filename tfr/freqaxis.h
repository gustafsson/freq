#ifndef TFRFREQAXIS_H
#define TFRFREQAXIS_H

#include <cuda_runtime.h> // defines __device__ and __host__
#include <math.h>

#include "msc_stdc.h"

namespace Tfr {

enum AxisScale {
    AxisScale_Linear,
    AxisScale_Logarithmic,
    AxisScale_Quefrency
};

/**
    FreqAxis defines a frequency axis in a way that can be used in a cuda
    kernel. The "switch (axis_scale)" lookup is considered 'fast enough' as
    the Chunk-to-Block scaling is assumed to be memory bound.

    FreqAxis is frequently used outside cuda kernels too.
*/
class FreqAxis
{
public:
    __device__ __host__ float getFrequency( unsigned fi ) const
    {
        return getFrequency( (float)fi );
    }


    __device__ __host__ float getFrequency( float fi ) const
    {
        switch (axis_scale)
        {
        case AxisScale_Linear:
            return f_min + fi*f_step;

        case AxisScale_Logarithmic:
            return f_min*exp2f( fi*log2f_step );

        case AxisScale_Quefrency:
            {
                float binmin = fs/f_min;
                float binmax = 2;
                float numbin = binmax-binmin;
                float bin = binmin + numbin*fi;
                return fs/bin;
            }
        default:
            return 0.f;
        }
    }


    __device__ __host__ unsigned getFrequencyIndex( float f ) const
    {
        float scalar = getFrequencyScalar( f );
        if (scalar < 0)
            scalar = 0;
        return (unsigned)(scalar + .5f);
    }


    __device__ __host__ float getFrequencyScalarNotClamped( float f ) const
    {
        float fi = 0;

        switch(axis_scale)
        {
        case AxisScale_Linear:
            fi = (f - f_min)/f_step;
            break;

        case AxisScale_Logarithmic:
            {
                float log2_f = log2f(f/f_min);

                fi = log2_f/log2f_step;
            }
            break;

        case AxisScale_Quefrency:
            fi = fs/f;
            break;
        }

        return fi;
    }

    __device__ __host__ float getFrequencyScalar( float f ) const
    {
        float fi = getFrequencyScalarNotClamped( f );
        if (fi > max_frequency_scalar) fi = max_frequency_scalar;
        return fi;
    }
//private:
//    friend class Chunk;
//    void FreqAxis() {} // Private default constructor. However, public default
//                       // copy constructor and assignment operator

    AxisScale axis_scale;

    float f_min;

    union {
        float log2f_step;
        float f_step;
        float fs;
    };

    float max_frequency_scalar;
};

} // namespace Tfr

#endif // TFRFREQAXIS_H
