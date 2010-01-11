#ifndef TRANSFORMDATA_H
#define TRANSFORMDATA_H

#include <boost/scoped_ptr.hpp>
#include "GpuCpuData.h"

class TransformData
{
public:
    /**
      rawData contains the wavelet transform rowwise.
      use as rawData[ sample + f_index*nSamples ];
      See getNearestCoeff for an example on how to find f_index.
    */
    boost::scoped_ptr<GpuCpuData<float> > transformData;
    float minHz, maxHz, sampleRate;
    float startTime;

    float timeInterval() const {    return nSamples()*sampleRate; }
    float endTime() const {         return startTime + timeInterval(); }
    unsigned nSamples() const {     return transformData->getNumberOfElements().width; }
    unsigned nFrequencies() const { return transformData->getNumberOfElements().height; }
    unsigned nChannels() const {    return transformData->getNumberOfElements().depth; }

    bool valid() const {
        return 0 != transformData->getSizeInBytes1D() &&
               0 != sampleRate &&
               minHz < maxHz;
    }

    float getNearestCoeff( float t, float f );
    unsigned getFrequencyIndex( float f ) const;
    float getFrequency( unsigned fi ) const;
};

#endif // TRANSFORMDATA_H

