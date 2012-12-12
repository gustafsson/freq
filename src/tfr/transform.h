#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "freqaxis.h"
#include "signal/intervals.h"

#include <boost/shared_ptr.hpp>

namespace Signal
{
    class MonoBuffer;
    typedef boost::shared_ptr<MonoBuffer> pMonoBuffer;
}

namespace Tfr
{

class TransformDesc;
class Chunk;
typedef boost::shared_ptr<Chunk> pChunk;

/**
  Examples of Transform implementations:
  "cwt.h"
  "stft.h"
  */
class Transform {
public:
    /**
      Virtual housekeeping.
      */
    virtual ~Transform() {}


    /**
     * @brief transformDesc
     * @return description used for this transform.
     */
    virtual const TransformDesc* transformDesc() const = 0;


    /**
      A Time-Frequency-Representation (Tfr) Transform takes a part of a signal
      (a Signal::Buffer) and transforms it into a part of a
      Time-Frequency-Representation (a Tfr::Chunk).

      Most Transforms works by transforming small sections of the input data
      independently. Thus to return a somewhat valid Chunk they need input data
      of at least the section size. An Fft plot though could take the entire
      signal and compute the transform exactly once. A Stft plot would prefer
      multiples of its window size, while an Cwt plot needs redundant samples
      before and after the valid output window.

      But a Transform is required to return something valid for any input
      size.
      */
    virtual pChunk operator()( Signal::pMonoBuffer b ) = 0;


    /**
      Well, transform a chunk back into a buffer.
      */
    virtual Signal::pMonoBuffer inverse( pChunk chunk ) = 0;
};

typedef boost::shared_ptr<Transform> pTransform;


/**
 * @brief The TransformDesc class represents a description to create a transform.
 */
class TransformDesc {
public:
    /**
      Virtual housekeeping.
      */
    virtual ~TransformDesc() {}


    /**
     * @brief createTransform instantiates a transform that uses the parameters in this description.
     * @return a newly created transform.
     */
    virtual pTransform createTransform() const = 0;


    /**
      At what time resolution (1/sample rate) it is meaningful to display the
      computed Chunks.
      */
    virtual float displayedTimeResolution( float FS, float hz ) const = 0;


    /**
      The frequency axis of chunks computed from a buffer with sample rate 'FS'.
      */
    virtual FreqAxis freqAxis( float FS ) const = 0;


    /**
      Returns the interval that could be validated using an interval of a given length.
      */
    //virtual Signal::Interval validInterval(Signal::pBuffer buffer) = 0;


    /**
      Returns the interval that would be required to yield a chunk with
      (getInterval() & I == I), i.e at least spanning I.
      */
    //virtual Signal::Interval requiredInterval(Signal::Interval I) = 0;


    /**
      Returns the next good chunk size for this type of transform (or the
      largest if there is no good chunk size larger than
      'current_valid_samples_per_chunk').
      */
    virtual unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const = 0;


    /**
      Returns the previously good chunk size for this type of transform (or the
      smallest if there is no good chunk size larger than
      'current_valid_samples_per_chunk').
      */
    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const = 0;


    /**
      Returns a string representation of this transform. Mainly used for debugging.
      */
    virtual std::string toString() const = 0;


    virtual bool operator==(const TransformDesc&) const = 0;
    bool operator!=(const TransformDesc& b) const { return !(*this == b); }
};

typedef boost::shared_ptr<TransformDesc> pTransformDesc;


} // namespace Tfr

#endif // TRANSFORM_H
