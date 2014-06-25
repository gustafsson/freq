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

Signal::IntervalType Chunk::
        offset(Signal::IntervalType sample, int f_index)
{
    if (sample >= nSamples())
        sample =  nSamples()-1;

    if (f_index >= (int)nScales())
        f_index =  nScales()-1;
    else if (f_index < 0)
        f_index = 0;

    switch(order) {
    case Order_row_major:
        return sample + f_index*nSamples();

    case Order_column_major:
        return sample*nScales() + f_index;

    default:
        return 0;
    }
}


ChunkElement Chunk::
        debug_getNearestCoeff( float t, float f )
{
    if (!valid())
        return ChunkElement(-FLT_MAX, -FLT_MAX);

    if ( t < 0 ) t = 0;

    unsigned s = (unsigned)(t*sample_rate+.5f);

    unsigned fi = freqAxis.getFrequencyIndex(f);

    return transform_data->getCpuMemory()[ offset(s, fi) ];
}


Signal::Interval Chunk::
        getInterval() const
{
    double scale = original_sample_rate/sample_rate;
    Signal::Interval I(
            std::floor((chunk_offset + first_valid_sample).asFloat() * scale + 0.5),
            std::floor((chunk_offset + first_valid_sample + n_valid_samples).asFloat() * scale + 0.5)
    );
    return I;
}


Signal::Interval Chunk::
        getCoveredInterval() const
{
    return getInterval();
}


} // namespace Tfr
