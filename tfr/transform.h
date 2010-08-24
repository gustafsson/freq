#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "signal/source.h"
#include "chunk.h"
#include "freqaxis.h"
#include <boost/shared_ptr.hpp>

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
      Each transform computes different frequency distributions. An instance of
      FreqAxis is used for translating frequencies to chunk indicies and vice
      versa. FreqAxis can be used within Cuda kernels.
      */
    FreqAxis freqInfo();


    /**
      Virtual housekeeping.
      */
    virtual ~Transform() {}
};
typedef boost::shared_ptr<Transform> pTransform;


} // namespace Tfr

#endif // TRANSFORM_H
