#include "transformdata.h"

#include <math.h>

float Transform_chunk::getNearestCoeff( float t, float f )
{
    if (!valid())
        return 0;

    if ( t < 0 ) t = 0;
    unsigned s = t*sampleRate+.5;
    if ( s >= nSamples() ) s=nSamples()-1;

    unsigned fi = getFrequencyIndex(f);

    return transformData->getCpuMemoryConst()[ fi*nSamples() + s ];
}

float Transform_chunk::getFrequency( unsigned fi ) const
{
    if (!valid())
        return 0;

    return exp(log(minHz) + (fi/(float)nFrequencies())*(log(maxHz)-log(minHz)));
}

unsigned Transform_chunk::getFrequencyIndex( float f ) const
{
    if (f<minHz) f=minHz;
    if (f>maxHz) f=maxHz;

    unsigned fi = round((log(f)-log(minHz))/(log(maxHz)-log(minHz))*nFrequencies());
    if (fi>nFrequencies()) fi = nFrequencies()-1;

    return fi;
}

