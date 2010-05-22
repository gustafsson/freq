#ifndef TFRCHUNK_H
#define TFRCHUNK_H

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "GpuCpuData.h"
#include "signal-samplesintervaldescriptor.h"

namespace Tfr {

struct Chunk
{
    Chunk();

    /**
      transform_data contains the complex wavelet transform rowwise.
      use as transform_data[ sample + f_index*nSamples ];
      See getNearestCoeff for an example on how to find f_index.
    */
    boost::scoped_ptr<GpuCpuData<float2> > transform_data;

    float min_hz, max_hz;

    /**
      chunk_offset is the start of the chunk, along the timeline, measured in
      samples.
      */
    unsigned chunk_offset;
    unsigned sample_rate;

    /**
      first_valid_sample is the first nonredundant column. first_valid_sample
      is also the number of redundant columns before the nonredundant data.
      */
    unsigned first_valid_sample;

    /**
      the number of nonredundant columns
      */
    unsigned n_valid_samples;

    float timeInterval() const {       return n_valid_samples/(float)sample_rate; }
    float startTime() const {          return (chunk_offset+first_valid_sample)/(float)sample_rate; }
    float endTime() const {            return startTime() + timeInterval(); }
    unsigned nSamples() const {        return transform_data->getNumberOfElements().width; }
    unsigned nScales() const {         return transform_data->getNumberOfElements().height; }
    unsigned nChannels() const {       return transform_data->getNumberOfElements().depth; }

    bool valid() const {
        return 0 != transform_data->getSizeInBytes1D() &&
               0 != sample_rate &&
               min_hz < max_hz;
    }

    float2 getNearestCoeff( float t, float f );
    unsigned getFrequencyIndex( float f ) const;
    float getFrequency( unsigned fi ) const;
    Signal::SamplesIntervalDescriptor::Interval getInterval() const;
};
typedef boost::shared_ptr< Chunk > pChunk;

} // namespace Tfr

#endif // TFRCHUNK_H
