#include "transformdata.h"

#include <math.h>

float TransformData::getNearestCoeff( float t, float f )
{
    if (!valid())
        return 0;

    if ( t < 0 ) t = 0;
    unsigned s = t*sampleRate+.5;
    if ( s >= nSamples() ) s=nSamples()-1;

    unsigned fi = round((log(f)-log(minHz))/(log(maxHz)-log(minHz))*nFrequencies());

    return transformData->getCpuMemoryConst()[ fi*nSamples() + s ];
}

float TransformData::getFrequency( unsigned fi ) const
{
    if (!valid())
        return 0;

    return exp(log(minHz) + (fi/(float)nFrequencies())*(log(maxHz)-log(minHz)));
}
