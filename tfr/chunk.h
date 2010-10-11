#ifndef TFRCHUNK_H
#define TFRCHUNK_H

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "GpuCpuData.h"
#include "signal/intervals.h"
#include "freqaxis.h"
#include "unsignedf.h"

namespace Tfr {


class Chunk
{
protected:
    Chunk( );
    virtual ~Chunk() {}

public:

    /**
      Each transform computes different frequency distributions. An instance of
      FreqAxis is used for translating frequencies to chunk indicies and vice
      versa. FreqAxis can be used within Cuda kernels.
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

    /**
      The highest and lowest frequency described by the chunk. Inclusive range.
      */
    float min_hz, max_hz;
    AxisScale axis_scale;

    /**
      chunk_offset is the start of the chunk, along the timeline, measured in
      signal samples, as described by 'first_valid_sample'.

        @see first_valid_sample
      */
    UnsignedF chunk_offset;


    /**
      first_valid_sample and n_valid_samples are such that the inverse
      transform would produce a buffer with this interval:

        Interval(
            chunk_offset + first_valid_sample,
            chunk_offset + first_valid_sample + n_valid_samples )

      @see chunk_offset
      */
    unsigned first_valid_sample;


    /**
        @see first_valid_sample
      */
    unsigned n_valid_samples;


    /**
      Columns per second.
      */
    float sample_rate;


    /**
      Original sample rate, the inverse will not necessarily produce a signal
      with this sample rate. It is used by Chunk::getInterval()
      */
    float original_sample_rate;

    float timeInterval() const {       return n_valid_samples/sample_rate; }
    float startTime() const {          return (chunk_offset+first_valid_sample)/sample_rate; }
    float endTime() const {            return startTime() + timeInterval(); }

    virtual unsigned nSamples() const {        return order==Order_row_major ? transform_data->getNumberOfElements().width : transform_data->getNumberOfElements().height; }
    virtual unsigned nScales() const {         return order==Order_row_major ? transform_data->getNumberOfElements().height: transform_data->getNumberOfElements().width;  }
    virtual unsigned nChannels() const {       return transform_data->getNumberOfElements().depth; }

    bool valid() const {
        return 0 != transform_data->getSizeInBytes1D() &&
               0 != sample_rate &&
               min_hz < max_hz &&
               (order == Order_row_major || order == Order_column_major);
    }

    float2 debug_getNearestCoeff( float t, float f );  /// For debugging


    /**

      */
    virtual Signal::Interval getInterval() const;
};
typedef boost::shared_ptr< Chunk > pChunk;

} // namespace Tfr

#endif // TFRCHUNK_H
