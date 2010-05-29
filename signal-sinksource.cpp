#include "signal-sinksource.h"
#include <boost/foreach.hpp>
#include <QMutexLocker>
#include <sstream>

using namespace std;

ostream& operator<<( ostream& s, const Signal::SamplesIntervalDescriptor::Interval& i)
{
    return s << "[" << i.first << ", " << i.last << "]";
}

namespace Signal {

SinkSource::
        SinkSource()
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


    unsigned FS = sample_rate();

    QMutexLocker l(&_mutex);

    if (!_cache.empty())
        BOOST_ASSERT(_cache.front()->sample_rate == b->sample_rate);

    std::stringstream ss;

    // Look among previous caches for buffers to remove
    for ( std::vector<pBuffer>::iterator itr = _cache.begin(); itr!=_cache.end(); itr++ )
    {
        const pBuffer& s = *itr;

        SamplesIntervalDescriptor toKeep = s->getInterval();
        toKeep -= b->getInterval();

        SamplesIntervalDescriptor toRemove = s->getInterval();
        toRemove &= b->getInterval();

        if (!toRemove.isEmpty()) {
            ss << "-" << s->getInterval() << " ";

            itr = _cache.erase(itr);

            BOOST_FOREACH( SamplesIntervalDescriptor::Interval i, toKeep.intervals() )
            {
                ss << "+" << i << " ";

                pBuffer n( new Buffer( i.first, i.last-i.first, FS));
                memcpy( n->waveform_data->getCpuMemory(),
                        s->waveform_data->getCpuMemory() + (i.first - s->sample_offset),
                        i.last-i.first );
                _cache.push_back( n );
            }
        }
    }

    _cache.push_back( b );

    if (!ss.str().empty())
    {
        ss << "+" << b->getInterval() << " ";
        TaskTimer("[%u, %u] merged: %s",
                     b->sample_offset,
                     b->sample_offset + b->number_of_samples(),
                     ss.str().c_str()).suppressTiming();
    }

    _expected_samples -= b->getInterval();

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
        sid |= s->getInterval();
    }

    return sid;
}

void SinkSource::
        add_expected_samples(SamplesIntervalDescriptor sid)
{    
    _expected_samples |= sid;
}

} // namespace Signal
