#ifndef TRANSFORMINVERSE_H
#define TRANSFORMINVERSE_H

/*
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

#include <boost/shared_ptr.hpp>
#include "signal-source.h"
#include "transform-chunk.h"
#include "filter.h"

typedef boost::shared_ptr<class Filter> pFilter;

class Transform_inverse
{
public:
    class Callback {
    public:
        virtual void transform_inverse_callback(Signal::pBuffer chunk)=0;
    };

    Transform_inverse( Signal::pSource waveform, Callback* callback, class Transform* temp_to_remove );

    void compute_inverse( float startt, float endt, pFilter filter );

    Signal::pBuffer         computeInverse( float start=0, float end=-1);
    Signal::pBuffer computeInverse( pTransform_chunk chunk, cudaStream_t stream=0 );
    Signal::pSource get_inverse_waveform();
    void      setInverseArea(float t1, float f1, float t2, float f2);
    void      recompute_filter(pFilter);

    void                merge_chunk(Signal::pBuffer r, pTransform_chunk transform);
    Signal::pBuffer     prepare_inverse(float start, float end);
    void      play_inverse();

    EllipsFilter built_in_filter;
private:
    Signal::pSource _original_waveform;
    Callback* _callback;
    Transform* _temp_to_remove;
    Signal::pSource _inverse_waveform;

    boost::shared_ptr<TaskTimer> filterTimer;
    float _t1, _f1, _t2, _f2;
};

#endif // TRANSFORMINVERSE_H
