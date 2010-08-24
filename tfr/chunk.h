#ifndef TFRCHUNK_H
#define TFRCHUNK_H

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "GpuCpuData.h"
#include "signal/samplesintervaldescriptor.h"
#include "freqaxis.h"

namespace Tfr {


// TODO perhaps Chunk should extend Signal::Bufer
class Chunk
{
public:
    Chunk( );

    /**
      A FreqAxis is used to translate frequencies to indicies and vice versa.
      */
    FreqAxis freqAxis();

    enum Order {
        Order_row_major,
        Order_column_major,
    } order;

    /**
      transform_data contains the complex transform.
      If order is Order_row_major they are stored rowwise such as:
       float2 value = transform_data[ sample + f_index*nSamples ];

      If order is Order_column_major they are stored columnwise such as:
       float2 value = transform_data[ sample*nScales + f_index ];

      See getNearestCoeff for an example on how to find f_index.
      'offset' can be used to give coordinates that takes order into account
      for computing the offset into the array.
    */
    boost::scoped_ptr<GpuCpuData<float2> > transform_data;


    unsigned offset(unsigned sample, unsigned f_index);


    float min_hz, max_hz;
    AxisScale axis_scale;

    /**
      chunk_offset is the start of the chunk, along the timeline, measured in
      samples.
      */
    unsigned chunk_offset;
    float sample_rate; // columns per second

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

    float2 debug_getNearestCoeff( float t, float f );  /// For debugging

    Signal::SamplesIntervalDescriptor::Interval getInterval() const;
};
typedef boost::shared_ptr< Chunk > pChunk;

} // namespace Tfr

#endif // TFRCHUNK_H
