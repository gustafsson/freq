#ifndef TFRFREQAXIS_H
#define TFRFREQAXIS_H

#include <cuda_runtime.h> // defines __device__ and __host__

namespace Tfr {

enum AxisScale {
    AxisScale_Linear,
    AxisScale_Logarithmic
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
        switch (axis_scale)
        {
        case AxisScale_Linear:
            return f_min + fi*f_step;

        case AxisScale_Logarithmic:
            return f_min*exp( fi*logf_step );

        default:
            return 0.f;
        }
    }

    __device__ __host__ unsigned getFrequencyIndex( float f ) const
    {
        unsigned fi = 0;

        switch(axis_scale)
        {
        case AxisScale_Linear:
            if (f > f_min)
            {
                fi = (unsigned)((f - f_min)/f_step +.5f);
                if (fi >= nscales) fi = nscales-1;
            }
            break;

        case AxisScale_Logarithmic:
            {
                float log_f = log(f/f_min);

                if (log_f > 1)
                {
                    fi = (unsigned)(log_f/logf_step +.5f);
                    if (fi >= nscales) fi = nscales-1;
                }
            }
            break;
        }

        return fi;
    }

private:
    friend class Chunk;

    AxisScale axis_scale;

    float f_min;

    union {
        float logf_step;
        float f_step;
    };

    unsigned nscales;
};

} // namespace Tfr

#endif // TFRFREQAXIS_H
