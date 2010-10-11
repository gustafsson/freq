#include "chunk.h"

#include "cwt.h"
#include "stft.h"

#include <math.h>

namespace Tfr {


Chunk::
        Chunk( )
:   min_hz(0),
    max_hz(0),
    axis_scale(AxisScale_Linear),
    chunk_offset(0),
    first_valid_sample(0),
    n_valid_samples(0),
    sample_rate(0)
{}

unsigned Chunk::
        offset(unsigned sample, unsigned f_index)
{
    if (sample >= nSamples())
        sample =  nSamples()-1;

    if (f_index >= nScales())
        f_index =  nScales()-1;

    switch(order) {
    case Order_row_major:
        return sample + f_index*nSamples();

    case Order_column_major:
        return sample*nScales() + f_index;

    default:
        return 0;
    }
}


float2 Chunk::
        debug_getNearestCoeff( float t, float f )
{
    if (!valid())
        return make_float2(0.f/0.f, 0.f/0.f);

    if ( t < 0 ) t = 0;

    unsigned s = (unsigned)(t*sample_rate+.5f);

    unsigned fi = freqAxis().getFrequencyIndex(f);

    return transform_data->getCpuMemoryConst()[ offset(s, fi) ];
}


FreqAxis Chunk::
        freqAxis()
{
    FreqAxis x;
    x.axis_scale = this->axis_scale;
    x.f_min = min_hz;

    switch (axis_scale)
    {
    case AxisScale_Logarithmic:
        x.max_frequency_scalar = nScales() - 1;
        x.log2f_step = (1.f/x.max_frequency_scalar) * (log2(max_hz)-log2(min_hz));
        break;

    case AxisScale_Linear:
        x.max_frequency_scalar = nScales()/2 - 1; // real transform, discard upper redundant half of spectra
        x.f_step = (1.f/x.max_frequency_scalar) * (max_hz - min_hz);
        break;

    default:
        throw std::invalid_argument("Unknown axis scale");
    }

    return x;
}


Signal::Interval Chunk::
        getInterval() const
{
    return Signal::Interval(
        chunk_offset + first_valid_sample,
        chunk_offset + first_valid_sample + n_valid_samples
    );
}

} // namespace Tfr
