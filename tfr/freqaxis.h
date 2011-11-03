#ifndef TFRFREQAXIS_H
#define TFRFREQAXIS_H

#ifndef FREQAXIS_CALL
#define FREQAXIS_CALL
#endif

#include <math.h>

#include "msc_stdc.h"

namespace Tfr {

enum AxisScale {
    AxisScale_Linear,
    AxisScale_Logarithmic,
    AxisScale_Quefrency,
    AxisScale_Unknown
};

/**
    FreqAxis defines a frequency axis in a way that can be used in a cuda
    kernel. The "switch (axis_scale)" lookup is considered 'fast enough' as
    the Chunk-to-Block scaling is assumed to be memory bound.

    FreqAxis is frequently used outside cuda kernels too.

    FreqAxis coordinates may be normalized frequency coordinates in the
    heightmap or matrix indices in a transform chunk.
*/
class FreqAxis
{
public:
    FreqAxis() : axis_scale(AxisScale_Unknown) {}


    bool operator==(const FreqAxis& b)
    {
        return axis_scale == b.axis_scale &&
                max_frequency_scalar == b.max_frequency_scalar &&
                min_hz == b.min_hz &&
                f_step == b.f_step;
    }


    /**
      Let max_frequency_scalar keep its default value 1 to create a normalized FreqAxis.
      */
    void setLinear( float fs, float max_frequency_scalar=1 )
    {
        this->axis_scale = AxisScale_Linear;

        this->max_frequency_scalar = max_frequency_scalar;
        float max_hz = fs/2;
        this->min_hz = 0;
        this->f_step = (1/max_frequency_scalar) * (max_hz - min_hz);
    }


    /**
      Let max_frequency_scalar keep its default value 1 to create a normalized
      FreqAxis.
      */
    void setLogarithmic( float min_hz_inclusive, float max_hz_inclusive, float max_frequency_scalar=1 )
    {
        this->axis_scale = AxisScale_Logarithmic;

        this->max_frequency_scalar = max_frequency_scalar;
        this->min_hz = min_hz_inclusive;
        this->f_step = log2f( max_hz_inclusive ) - log2f( min_hz_inclusive );
        this->f_step /= max_frequency_scalar;
    }


    /**
      Create a normalized FreqAxis for a cepstragram heightmap.
      */
    void setQuefrencyNormalized( float fs, float window_size )
    {
        this->axis_scale = AxisScale_Quefrency;

        this->max_frequency_scalar = 1;
        this->min_hz = 2*fs/window_size;
        this->f_step = fs/min_hz;
    }


    /**
      Create a normal FreqAxis for cepstragram data.
      */
    void setQuefrency( float fs, float window_size )
    {
        this->axis_scale = AxisScale_Quefrency;

        this->max_frequency_scalar = window_size/2;
        this->min_hz = fs/max_frequency_scalar;
        this->f_step = -1;
    }


    /**
      Translates FreqAxis coordinates to 'Hz'. FreqAxis coordinates may be
      normalized frequency coordinates in the heightmap or matrix indices in
      a transform chunk.
      @see getFrequencyScalar
      */
    FREQAXIS_CALL float getFrequency( unsigned fi ) const
    {
        return getFrequency( (float)fi );
    }


    /**
      Translates FreqAxis coordinates to 'Hz'.
      @see getFrequencyScalar
      */
    FREQAXIS_CALL float getFrequency( float fi ) const
    {
        switch (axis_scale)
        {
        case AxisScale_Linear:
            return min_hz + fi*f_step;

        case AxisScale_Logarithmic:
            return min_hz*exp2f( fi*f_step );

        case AxisScale_Quefrency:
            {
                if (f_step>0) // normalized
                {
                    float fs = max_frequency_scalar*min_hz*f_step;
                    float binmin = fs/min_hz;
                    float binmax = 2;
                    float numbin = binmax-binmin;
                    float bin = binmin + numbin*fi;
                    return fs/bin;
                }
                else
                {
                    float fs = max_frequency_scalar*min_hz;
                    return fs/fi;
                }
            }
        default:
            return 0.f;
        }
    }


    /// @see getFrequencyScalar
    FREQAXIS_CALL unsigned getFrequencyIndex( float hz ) const
    {
        float scalar = getFrequencyScalar( hz );
        if (scalar < 0)
            scalar = 0;
        return (unsigned)(scalar + .5f);
    }


    /// @see getFrequencyScalar
    FREQAXIS_CALL float getFrequencyScalarNotClamped( float hz ) const
    {
        float fi = 0;

        switch(axis_scale)
        {
        case AxisScale_Linear:
            fi = (hz - min_hz)/f_step;
            break;

        case AxisScale_Logarithmic:
            {
                float log2_f = log2f(hz/min_hz);

                fi = log2_f/f_step;
            }
            break;

        case AxisScale_Quefrency:
            {
                if (f_step>0)
                {
                    float fs = max_frequency_scalar*min_hz*f_step;
                    float binmin = fs/min_hz;
                    float binmax = 2;
                    float numbin = binmax-binmin;
                    fi = (fs/hz - binmin)/numbin;
                }
                else
                {
                    float fs = max_frequency_scalar*min_hz;
                    fi = fs/hz;
                }
            }
            break;
        default:
            break;
        }

        return fi;
    }


    /**
      Translate 'hz' to FreqAxis coordinates. FreqAxis coordinates may be
      normalized frequency coordinates in the heightmap or matrix indices in
      a transform chunk.
      */
    FREQAXIS_CALL float getFrequencyScalar( float hz ) const
    {
        float fi = getFrequencyScalarNotClamped( hz );
        if (fi > max_frequency_scalar) fi = max_frequency_scalar;
        return fi;
    }


    /**
      The highest frequency along this frequency axis.
      */
    float max_hz() const // inclusive
    {
        switch(axis_scale)
        {
        case AxisScale_Linear:
        case AxisScale_Logarithmic:
            return getFrequency(max_frequency_scalar);
        case AxisScale_Quefrency:
            if (0<f_step)
                return max_frequency_scalar*min_hz*f_step/2;
            else
                return max_frequency_scalar*min_hz/2;
        default:
            return 0;
        }
    }


    AxisScale axis_scale;
    /// the lowest frequency in Hertz that inclusive
    float min_hz;
    /// differnet usages based on axis_scale
    float f_step;
    /// highest value that will be returned from getFrequencyScalar or getFrequencyIndex
    float max_frequency_scalar;
};

} // namespace Tfr

#endif // TFRFREQAXIS_H
