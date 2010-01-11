#ifndef TRANSFORMCHUNK_H
#define TRANSFORMCHUNK_H

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "GpuCpuData.h"

typedef boost::shared_ptr<class Transform_chunk> pTransform_chunk;

class Transform_chunk
{
public:
    /**
      transform_data contains the wavelet transform rowwise.
      use as transform_data[ sample + f_index*nSamples ];
      See getNearestCoeff for an example on how to find f_index.
    */
    boost::scoped_ptr<GpuCpuData<float> > transform_data;

    float minHz, maxHz, sampleRate;
    unsigned sampleOffset;

    float timeInterval() const {    return nSamples()/sampleRate; }
    float startTime() const {       return sampleOffset/sampleRate; }
    float endTime() const {         return startTime() + timeInterval(); }
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

#endif // TRANSFORMCHUNK_H
