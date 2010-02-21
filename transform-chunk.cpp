#include "transform-chunk.h"

#include <math.h>

Transform_chunk::Transform_chunk()
:   min_hz(0),
    max_hz(0),
    chunk_offset(0),
    sample_rate(0),
    first_valid_sample(0),
    n_valid_samples(0),
    modified(false)
{}

float2 Transform_chunk::getNearestCoeff( float t, float f )
{
    if (!valid())
        return make_float2(0,0);

    if ( t < 0 ) t = 0;
    unsigned s = (unsigned)(t*sample_rate+.5);
    if ( s >= nSamples() ) s=nSamples()-1;

    unsigned fi = getFrequencyIndex(f);

    return transform_data->getCpuMemoryConst()[ fi*nSamples() + s ];
}

float Transform_chunk::getFrequency( unsigned fi ) const
{
    if (!valid())
        return 0;

    return exp(log(min_hz) + (fi/(float)nScales())*(log(max_hz)-log(min_hz)));
}

unsigned Transform_chunk::getFrequencyIndex( float f ) const
{
    if (f<min_hz) f=min_hz;
    if (f>max_hz) f=max_hz;

    unsigned fi = (unsigned)((log(f)-log(min_hz))/(log(max_hz)-log(min_hz))*nScales());
    if (fi>nScales()) fi = nScales()-1;

    return fi;
}
