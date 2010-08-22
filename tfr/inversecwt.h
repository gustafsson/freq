#ifndef TFRINVERSECWT_H
#define TFRINVERSECWT_H

#include "tfr/chunk.h"
#include "signal/source.h"
#include <boost/shared_ptr.hpp>
#include "tfr/filter.h"

namespace Tfr {
/**
TODO: Obsolete text, remove or rewrite:

Transform_inverse produces a streaming inverse of the transform in chunks of
length dt.
1 Select chunk of original waveform, elapsing from t-wt to t+dt+wt. This
  includes some data, wt, before t and after t+dt, that is required to compute
  the transform properly.
2 Create the wavelett transform, with "say 40" scales per octave.
3 Apply the filter chain.
3.1 The filter chain might request transforms previously computed or transforms
    not yet computed. These need to be computed on spot on a cache-miss.
4 Compute the inverse elapsing from t to t+dt, discarding wt before and after.
5 Send inverse to callback function. The callback function is responsible for
  caching and may block execution if it is fed with data too fast.
6 Start over, overwrite the transform of the chunk furthest away from [t,t+dt).
  */
class InverseCwt
{
public:
    InverseCwt(cudaStream_t stream=0);

    Signal::pBuffer operator()(Tfr::Chunk&);

    pFilter filter;
private:
    cudaStream_t _stream;
};

} // namespace Tfr

#endif // TFRINVERSECWT_H
