#include "buffersource.h"

namespace Signal {


BufferSource::
        BufferSource( pBuffer waveform )
:    channel(0)
{
    setBuffer( waveform );
}


void BufferSource::
        setBuffer( pBuffer waveform )
{
    if (0==waveform || 0==waveform->number_of_samples())
    {
        _waveforms.resize(1);
        _waveforms[0] = waveform;
        return;
    }

    cudaExtent sz = waveform->waveform_data()->getNumberOfElements();
    unsigned number_of_samples = sz.width;
    unsigned channels = sz.height;
    _waveforms.resize(channels);
    if (0==channels)
        ;
    else if (1==channels)
        _waveforms[0] = waveform;
    else for (unsigned c=0; c<channels; ++c)
    {
        pBuffer w(new Buffer(waveform->sample_offset, number_of_samples, waveform->sample_rate));
        memcpy( w->waveform_data()->getCpuMemory(),
                waveform->waveform_data()->getCpuMemory() + c*number_of_samples,
                w->waveform_data()->getSizeInBytes1D() );
        _waveforms[c] = w;
    }
}


pBuffer BufferSource::
        read( const Interval& I )
{
    BOOST_ASSERT( channel < num_channels() );

    const Interval& myInterval = _waveforms[channel]->getInterval();
    if (Intervals(I.first, I.first+1) & myInterval)
    {
        return _waveforms[channel];
    }

    return zeros((Intervals(I) - myInterval).getInterval());
}


float BufferSource::
        sample_rate()
{
    return _waveforms[0]->sample_rate;
}


long unsigned BufferSource::
        number_of_samples()
{
    return _waveforms[0]->number_of_samples();
}

} // namespace Signal
