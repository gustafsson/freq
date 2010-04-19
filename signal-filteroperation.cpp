#include "signal-filteroperation.h"

namespace Signal {

FilterOperation::
FilterOperation(Tfr::pFilter filter)
:   _filter( filter )
{
}

pBuffer FilterOperation::
read( unsigned firstSample, unsigned numberOfSamples )
{
    if (firstSample < _wavelet_std_samples ) firstSample = 0;
    else firstSample -= _wavelet_std_samples;

    pBuffer b = _source->read( firstSample, numberOfSamples + 2*_wavelet_std_samples );

    pChunk c = _cwt( b );
    _filter( c );
    Tfr::InverseCwt inverse_cwt;
    pBuffer r = inverse_cwt( c );

    _previous_chunk = c;

    return r;
}

} // namespace Signal
