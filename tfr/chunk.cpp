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
    sample_rate(0),
    first_valid_sample(0),
    n_valid_samples(0)
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
    }
}


float2 Chunk::
        debug_getNearestCoeff( float t, float f )
{
    if (!valid())
        return make_float2(0,0);

    if ( t < 0 ) t = 0;
    unsigned s = (unsigned)(t*sample_rate+.5);
    if ( s >= nSamples() ) s=nSamples()-1;

    unsigned fi = freqAxis().getFrequencyIndex(f);

    return transform_data->getCpuMemoryConst()[ fi*nSamples() + s ];
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
        x.nscales = nScales();
        x.logf_step = (1.f/(nScales()-1)) * (log(max_hz)-log(min_hz));
        break;

    case AxisScale_Linear:
        x.nscales = nScales() / 2; // just discard half of the scales
        x.f_step = (1.f/(x.nscales-1)) * (max_hz - min_hz);
        break;

    default:
        throw std::invalid_argument("Unknown axis scale");
    }

    return x;
}


Signal::SamplesIntervalDescriptor::Interval Chunk::
        getInterval() const
{
    Signal::SamplesIntervalDescriptor::Interval i = {
        chunk_offset + first_valid_sample,
        chunk_offset + first_valid_sample + n_valid_samples
    };
    return i;
}

} // namespace Tfr
