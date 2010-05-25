#include "signal-sinksource.h"
#include <boost/foreach.hpp>
#include <QMutexLocker>
#include <sstream>

namespace Signal
{
SinkSource::
        SinkSource(pSource s)
:   Operation(s)
{

}

void SinkSource::
        put( pBuffer b )
{
    /* CANONICAL {
        QMutexLocker l(&_mutex);

        // Simply remove previous overlapping buffer, don't bother merging.
        BOOST_FOREACH( pBuffer& s, _cache) {
        {
            if (!s)
                s = b;
            else if (s->sample_offset < b->sample_offset+b->number_of_samples() && s->sample_offset + s->number_of_samples() > b->sample_offset)
            {
                s = b;
                b.reset(); // If copied more than once, set others to 0
            }
        }
    }

    _cache.push_back( b );
    */

    if (!_cache.empty())
        BOOST_ASSERT(_cache.front()->sample_rate == b->sample_rate);

    SamplesIntervalDescriptor sid( b->sample_offset, b->sample_offset + b->number_of_samples());
    sid -= samplesDesc();

    unsigned FS = sample_rate();

    {
        QMutexLocker l(&_mutex);

        std::stringstream ss;

        // Merge
        BOOST_FOREACH( SamplesIntervalDescriptor::Interval i, sid.intervals() )
        {
            if (i.first == b->sample_offset && i.last == b->sample_offset + b->number_of_samples())
            {
                _cache.push_back( b );
                break;
            }

            ss << " [" << i.first <<", "<<i.last<<"]";

            pBuffer n( new Buffer( i.first, i.last-i.first, FS));
            memcpy( n->waveform_data->getCpuMemory(),
                    b->waveform_data->getCpuMemory() + (i.first - b->sample_offset),
                    i.last-i.first );
            _cache.push_back( n );
        }

        if (!ss.str().empty())
        {
            TaskTimer("Merged buffer [%u, %u] in chunks: %s",
                         b->sample_offset,
                         b->sample_offset + b->number_of_samples(),
                         ss.str().c_str()).suppressTiming();
        }
    }

    _expected_samples -= sid;

    _expected_samples.print("SinkSource _expected_samples");
}

void SinkSource::
        reset()
{
    _cache.clear();
    _expected_samples = SamplesIntervalDescriptor();
}

pBuffer SinkSource::
        read( unsigned firstSample, unsigned numberOfSamples )
{
    {
        QMutexLocker l(&_mutex);

        BOOST_FOREACH( const pBuffer& s, _cache) {
            if (s->sample_offset <= firstSample && s->sample_offset + s->number_of_samples() > firstSample )
            {
                TaskTimer(TaskTimer::LogVerbose, "Reading [%u, %u]", s->sample_offset, s->sample_offset + s->number_of_samples()).suppressTiming();
                return s;
            }
        }
    }

    if (_source) {
        pBuffer b = _source->read( firstSample, numberOfSamples );
        put(b);
        return b;
    }

    TaskTimer(TaskTimer::LogVerbose, "SILENT!").suppressTiming();
    pBuffer b( new Buffer(firstSample, numberOfSamples, sample_rate()));
    memset( b->waveform_data->getCpuMemory(), 0, b->waveform_data->getSizeInBytes1D() );
    return b;
}

unsigned SinkSource::
        sample_rate()
{
    QMutexLocker l(&_mutex);

    if (_cache.empty())
        return (unsigned)-1;
    return _cache.front()->sample_rate;
}

unsigned SinkSource::
        number_of_samples()
{
    unsigned n = 0;

    QMutexLocker l(&_mutex);

    BOOST_FOREACH( const pBuffer& s, _cache) {
        n += s->number_of_samples();
    }

    return n;
}

pBuffer SinkSource::
        first_buffer()
{
    QMutexLocker l(&_mutex);
    if (_cache.empty())
        return pBuffer();
    return _cache.front();
}

bool SinkSource::empty()
{
    QMutexLocker l(&_mutex);

    return _cache.empty();
}

unsigned SinkSource::size()
{
    QMutexLocker l(&_mutex);
    return _cache.size();
}

SamplesIntervalDescriptor SinkSource::
        samplesDesc()
{
    QMutexLocker l(&_mutex);

    SamplesIntervalDescriptor sid;

    BOOST_FOREACH( const pBuffer& s, _cache) {
        sid |= SamplesIntervalDescriptor( s->sample_offset, s->sample_offset + s->number_of_samples() );
    }

    return sid;
}

void SinkSource::
        add_expected_samples(SamplesIntervalDescriptor sid)
{    
    _expected_samples |= sid;

    BOOST_FOREACH( SamplesIntervalDescriptor::Interval i, sid.intervals()) {
        BOOST_FOREACH( pBuffer& s, _cache) {
            if (s->sample_offset + s->number_of_samples() > i.first && s->sample_offset < i.last) {
                _expected_samples |= SamplesIntervalDescriptor(s->sample_offset, s->sample_offset + s->number_of_samples());
                s.reset();
            }
        }
    }

}

} // namespace Signal
