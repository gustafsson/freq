#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "chunk.h"
#include "freqaxis.h"

#include "signal/source.h"

namespace Tfr
{

/**
  Examples of Transform implementations:
  "cwt.h"
  "stft.h"
  */
class Transform {
public:
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
    virtual pChunk operator()( Signal::pBuffer b ) = 0;


    /**
      Well, transform a chunk back into a buffer.
      */
    virtual Signal::pBuffer inverse( pChunk chunk ) = 0;


    /**
      At what time resolution (1/sample rate) it is meaningful to display the
      computed Chunks.
      */
    virtual float displayedTimeResolution() { return 1/44100.f; }


    /**
      At what frequency resolution (smallest f(i+1)/f(i)) it is meaningful to
      display the computed Chunks.
      */
    virtual float displayedFrequencyResolution() { return 1.01f; }


    /**
      The frequency axis of chunks computed from a buffer with sample rate 'FS'.
      */
    virtual FreqAxis freqAxis( float FS ) = 0;


    /**
      Virtual housekeeping.
      */
    virtual ~Transform() {}
};
typedef boost::shared_ptr<Transform> pTransform;


} // namespace Tfr

#endif // TRANSFORM_H
