#include "chunk.h"

#include <math.h>
#include <float.h>

namespace Tfr {


Chunk::
        Chunk( Order order )
:   order(order),
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
        return make_float2(-FLT_MAX, -FLT_MAX);

    if ( t < 0 ) t = 0;

    unsigned s = (unsigned)(t*sample_rate+.5f);

    unsigned fi = freqAxis.getFrequencyIndex(f);

    return transform_data->getCpuMemoryConst()[ offset(s, fi) ];
}


Signal::Interval Chunk::
        getInversedInterval() const
{
    return Signal::Interval(
        chunk_offset + first_valid_sample,
        chunk_offset + first_valid_sample + n_valid_samples
    );
}


Signal::Interval Chunk::
        getInterval() const
{
    double scale = original_sample_rate/sample_rate;
    return Signal::Interval(
            std::floor((chunk_offset + first_valid_sample).asFloat() * scale + 0.5),
            std::floor((chunk_offset + first_valid_sample + n_valid_samples).asFloat() * scale + 0.5)
    );
}



} // namespace Tfr
