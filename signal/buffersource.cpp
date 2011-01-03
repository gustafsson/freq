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
    if (Intervals(I.first, I.first+1) & _waveform->getInterval())
        return _waveform;

    return zeros(I);
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
