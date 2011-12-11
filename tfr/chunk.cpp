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


Signal::Interval Chunk::
        getCoveredInterval() const
{
    double scale = original_sample_rate/sample_rate;
    Signal::Interval I(
            std::floor((chunk_offset + .5f).asFloat() * scale + 0.5),
            std::floor((chunk_offset + nSamples() - .5f).asFloat() * scale + 0.5)
    );

    if (0 == chunk_offset)
    {
        I.first = 0;
        I.last = std::floor((nSamples() - .5f) * scale + 0.5);
    }

    return I;
}


} // namespace Tfr
