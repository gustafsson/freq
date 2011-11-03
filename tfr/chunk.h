#ifndef TFRCHUNK_H
#define TFRCHUNK_H

#include "signal/intervals.h"
#include "chunkdata.h"

// gpusmisc
#include "freqaxis.h"
#include "unsignedf.h"
#include "datastorage.h"

// boost
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

// std
#include <complex>

namespace Tfr {


class Chunk
{
protected:
    enum Order {
        Order_row_major,
        Order_column_major,
    } order;

    Chunk( Order order );
    virtual ~Chunk() {}

public:

    /**
      Each transform computes different frequency distributions. An instance of
      FreqAxis is used for translating frequencies to chunk indicies and vice
      versa. FreqAxis can be used within Cuda kernels.
      */
    FreqAxis freqAxis;

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
    ChunkData::Ptr transform_data;


    unsigned offset(unsigned sample, unsigned f_index);


    /**
      chunk_offset is the start of the chunk, along the timeline. Measured in
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

      This interval is returned by getInversedInterval.

      @see chunk_offset
      @see getInterval
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

    /**
      The highest and lowest frequency described by the chunk. Inclusive range.
      */
    float minHz() const {              return freqAxis.min_hz; }
    float maxHz() const {              return freqAxis.max_hz(); }

    float timeInterval() const {       return n_valid_samples/sample_rate; }
    float startTime() const {          return (chunk_offset+first_valid_sample)/sample_rate; }
    float endTime() const {            return startTime() + timeInterval(); }

    virtual unsigned nSamples() const {        return order==Order_row_major ? transform_data->getNumberOfElements().width : transform_data->getNumberOfElements().height; }
    virtual unsigned nScales() const {         return order==Order_row_major ? transform_data->getNumberOfElements().height: transform_data->getNumberOfElements().width;  }
    virtual unsigned nChannels() const {       return transform_data->getNumberOfElements().depth; }

    bool valid() const {
        return 0 != transform_data->getSizeInBytes1D() &&
               0 != sample_rate &&
               minHz() < maxHz() &&
               (order == Order_row_major || order == Order_column_major);
    }

    std::complex<float> debug_getNearestCoeff( float t, float f );  /// For debugging


    /**
      Returns the interval that would be produced by the inverse transform.

        Interval(
            chunk_offset + first_valid_sample,
            chunk_offset + first_valid_sample + n_valid_samples )
      */
    virtual Signal::Interval getInversedInterval() const;


    /**
      Returns an equivalent interval in the original sample rate.

        Interval(
            getInversedInterval().first*original_sample_rate/sample_rate,
            getInversedInterval().last*original_sample_rate/sample_rate)
      */
    virtual Signal::Interval getInterval() const;

};
typedef boost::shared_ptr< Chunk > pChunk;

} // namespace Tfr

#endif // TFRCHUNK_H
