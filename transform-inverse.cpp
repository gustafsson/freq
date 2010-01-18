#include "transform-inverse.h"
#include "transform-chunk.h"

Transform_inverse::Transform_inverse(pWaveform waveform, Callback* callback)
:   _waveform(waveform),
    _callback(callback)
{
}

void Transform_inverse::compute_inverse( float startt, float endt, pFilter filter )
{
    float chunksize=.1; // seconds
    float waveletsize=.05; // extra

    filter.get();

    for (float t=startt; t<endt; t+=chunksize) {
        float t1 = t-waveletsize;
        if (t1<0) t1=0;
        float t2 = t+chunksize+waveletsize;
        if (t2>endt) t2=endt;

        /*
        pWaveform_chunk waveform = _waveform->getChunk(t1,t2);
        pTransform_chunk transform = morlet( waveform );
        transform->getTransform_chunk(t1, t2);

        filter(transform);

        Waveform_chunk inverse = morlet_inverse( transform );

        _callback->transform_inverse_callback( inverse );
        */
    }
}
