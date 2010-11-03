#include "buffersource.h"

namespace Signal {


BufferSource::
        BufferSource( pBuffer waveform )
:    _waveform(waveform)
{
}


pBuffer BufferSource::
        read( const Interval& I )
{
    const Interval& myInterval = _waveform->getInterval();
    if (Intervals(I.first, I.first+1) & myInterval)
        return _waveform;

    return zeros((Intervals(I) - myInterval).getInterval());
}


float BufferSource::
        sample_rate()
{
    return _waveform->sample_rate;
}


long unsigned BufferSource::
        number_of_samples()
{
    return _waveform->number_of_samples();
}

} // namespace Signal
