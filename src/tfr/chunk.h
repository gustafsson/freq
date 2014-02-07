#ifndef TFRCHUNK_H
#define TFRCHUNK_H

#include "signal/intervals.h"
#include "chunkdata.h"

// gpusmisc
#include "freqaxis.h"
#include "unsignedf.h"

// boost
#include <boost/shared_ptr.hpp>

// std
#include <complex>

namespace Tfr {


class Chunk
{
public:
    enum Order {
        Order_row_major,
        Order_column_major
    } order;

protected:
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


    Signal::IntervalType offset(Signal::IntervalType sample, int f_index);


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
            chunk_offset + first_valid_sample + n_valid_samples ) * original_sample_rate/sample_rate

      This interval is returned by getInterval.


        Example
        window size = 40 => alignment = 40
        overlap = 75% => increment = 10
        averaging = 1

        Time series samples
        0    10   20   30   40   50   60   70   80   90   100  110  120
        |----|----|----|----|----|----|----|----|----|----|----|----|

        STFT overlapping windows
        [-------(-0-)------][-------(-4-)------][-------(-8-)------]
             [-------(-1-)------][-------(-5-)------]
                  [-------(-2-)------][-------(-6-)------]
                       [-------(-3-)------][-------(-7-)------]

        chunk_offset = 0
        the first chunk represents the frequency around sample index 20.
        Measured in steps of increment*averaging. Start at sample number 15.

        first_valid_sample = 3
        when computing the inverse, window number 3 is the first window with full support
        (note that "first_valid_sample" only makes sense if the transform is invertible)

        valid samples = 3
        window number 3, 4 and 5 have full support

        getInterval = [30, 90)
        getCoveredInterval = [20, 100)

        STFT disjoint windows
        0    10   20   30   40   50   60   70   80   90   100  110  120
        |----|----|----|----|----|----|----|----|----|----|----|----|
        [---------0--------][---------1--------][---------2--------]

        chunk_offset = 0
        first_valid_sample = 0
        valid samples = 3

        getInterval = [0, 120)
        getCoveredInterval = [20, 100)

      @see chunk_offset
      @see getInterval
      */
    int first_valid_sample;


    /**
        @see first_valid_sample
      */
    int n_valid_samples;


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
    float startTime() const {          return ((chunk_offset+first_valid_sample)/sample_rate).asFloat(); }
    float endTime() const {            return startTime() + timeInterval(); }

    virtual unsigned nSamples() const {        return order==Order_row_major ? transform_data->size().width : transform_data->size().height; }
    virtual unsigned nScales() const {         return order==Order_row_major ? transform_data->size().height: transform_data->size().width;  }
    virtual unsigned nChannels() const {       return transform_data->size().depth; }

    bool valid() const {
        return 0 != transform_data->numberOfBytes() &&
               0 != sample_rate &&
               minHz() < maxHz() &&
               (order == Order_row_major || order == Order_column_major);
    }

    std::complex<float> debug_getNearestCoeff( float t, float f );  /// For debugging


    /**
      Returns an equivalent interval in the original sample rate.

        Interval(
            chunk_offset + first_valid_sample,
            chunk_offset + first_valid_sample + n_valid_samples )

        But rescaled by original_sample_rate/sample_rate
      */
    virtual Signal::Interval getInterval() const;


    /**
      Returns an equivalent interval in the original sample rate.

        Interval(
            chunk_offset*original_sample_rate/sample_rate ,
            (chunk_offset + nSamples())*original_sample_rate/sample_rate )
      */
    virtual Signal::Interval getCoveredInterval() const;
};
typedef boost::shared_ptr< Chunk > pChunk;

} // namespace Tfr

#endif // TFRCHUNK_H
