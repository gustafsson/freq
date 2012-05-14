#include "buffersource.h"

#include "sawe/configuration.h"

#include <boost/foreach.hpp>

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

    DataStorageSize sz = waveform->waveform_data()->size();
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

    return zeros((Intervals(I) - myInterval).fetchFirstInterval());
}


float BufferSource::
        sample_rate()
{
    return _waveforms[0]->sample_rate;
}


void BufferSource::
        set_sample_rate( float fs )
{
    bool changed = false;
    for (unsigned i=0; i< _waveforms.size(); ++i )
    {
        pBuffer b = _waveforms[i];

        changed |= b->sample_rate != fs;

        b->sample_rate = fs;
    }

    invalidate_samples( Signal::Interval::Interval_ALL );
}


Signal::IntervalType BufferSource::
        number_of_samples()
{
    return _waveforms[0]->number_of_samples();
}


unsigned BufferSource::
        num_channels()
{
    if (Sawe::Configuration::mono())
        return _waveforms.size() ? 1 : 0;
    else
        return _waveforms.size();
}


void BufferSource::
        set_channel(unsigned c)
{
    BOOST_ASSERT( c < num_channels() );

    channel = c;
}


} // namespace Signal
