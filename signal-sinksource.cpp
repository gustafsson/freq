#include "signal-sinksource.h"
#include <boost/foreach.hpp>
#include <QMutexLocker>
#include <sstream>

static const bool D = false;

using namespace std;

namespace Signal {

SinkSource::
        SinkSource( AcceptStrategy a )
:   _acceptStrategy( a )
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

    switch (_acceptStrategy)
    {
    case AcceptStrategy_ACCEPT_ALL:
        merge(b);
        break;
    case AcceptStrategy_ACCEPT_EXPECTED_ONLY:
        {
            SamplesIntervalDescriptor expected = expected_samples();
            if ((SamplesIntervalDescriptor(b->getInterval()) - expected).isEmpty())
                // This entire buffer was expected, merge
                merge(b);
            else
            {
                SamplesIntervalDescriptor sid = expected & b->getInterval();

                // Signal::Source have readFixedLength which can divide a
                // Buffer into sections if it is presented as a Source which
                // is accomplished by a SinkSource.
                SinkSource ss(AcceptStrategy_ACCEPT_ALL);

                // shortcut to _cache instead of put which would have
                // accomplished the same
                ss._cache.push_back( b );

                BOOST_FOREACH( const SamplesIntervalDescriptor::Interval i, sid.intervals() )
                {
                    pBuffer s = ss.readFixedLength( i.first, i.last-i.first );
                    merge( s );
                }
            }
        }
        break;
    default:
        BOOST_ASSERT(false);
        break;
    }
}

static bool bufferLessThan(const pBuffer& a, const pBuffer& b)
{
    return a->sample_offset < b->sample_offset;
}

// Smallest power of two greater than x
static unsigned int
spo2g(register unsigned int x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return(x+1);
}

// Largest power of two smaller than x
static unsigned int
lpo2s(register unsigned int x)
{
    return spo2g(x-1)>>1;
}

void SinkSource::
        selfmerge()
{
	{
		QMutexLocker l(&_cache_mutex);

		if (_cache.empty())
			return;
	}

    //TaskTimer tt("SinkSource::selfmerge");
    //samplesDesc().print("selfmerged start");
    //tt.info("_cache.size()=%u", _cache.size());

    SamplesIntervalDescriptor sid = samplesDesc();
	std::vector<pBuffer> new_cache;

	BOOST_FOREACH( SamplesIntervalDescriptor::Interval i, sid.intervals() )
	{
		for(unsigned L=0; i.first < i.last; i.first+=L)
		{
			L = lpo2s( i.last - i.first + 1);
			if (L<(1<<13))
				L = i.last-i.first;
			new_cache.push_back(readFixedLength( i.first, L ));
		}
	}

	{
		QMutexLocker l(&_cache_mutex);
	    _cache = new_cache;
	}

    //samplesDesc().print("selfmerged finished");
    //tt.info("_cache.size()=%u", _cache.size());
}

void SinkSource::
        merge( pBuffer bp )
{
    const Buffer& b = *bp;
    unsigned FS = sample_rate();

    QMutexLocker cache_locker(&_cache_mutex);

    if (!_cache.empty())
        BOOST_ASSERT(_cache.front()->sample_rate == b.sample_rate);

    std::stringstream ss;

    // REMOVE caches that become outdated by this new buffer 'b'
    for ( std::vector<pBuffer>::iterator itr = _cache.begin(); itr!=_cache.end(); )
    {
        const pBuffer ps = *itr;
        const Buffer& s = *ps;

        SamplesIntervalDescriptor toKeep = s.getInterval();
        toKeep -= b.getInterval();

        SamplesIntervalDescriptor toRemove = s.getInterval();
        toRemove &= b.getInterval();

        if (!toRemove.isEmpty()) {
            if(D) ss << " -" << s.getInterval();

            itr = _cache.erase(itr); // Note: 'pBuffer s' stores a copy for the scope of the for-loop

            BOOST_FOREACH( SamplesIntervalDescriptor::Interval i, toKeep.intervals() )
            {
                if(D) ss << " +" << i;

                pBuffer n( new Buffer( i.first, i.last-i.first, FS));
                memcpy( n->waveform_data->getCpuMemory(),
                        s.waveform_data->getCpuMemory() + (i.first - s.sample_offset),
                        n->waveform_data->getSizeInBytes1D() );
                itr = _cache.insert(itr, n );
                itr++; // Move past inserted element
            }
        } else {
            itr++;
        }
    }

    pBuffer n( new Buffer( b.sample_offset, b.number_of_samples(), b.sample_rate));
    memcpy( n->waveform_data->getCpuMemory(),
            b.waveform_data->getCpuMemory(),
            b.waveform_data->getSizeInBytes1D());
    _cache.push_back( n );
    cache_locker.unlock(); // done with _cache, samplesDesc() below needs the lock

    if(D) if (!ss.str().empty())
    {
        ss << " +" << n->getInterval();
        TaskTimer("M:%s", ss.str().c_str()).suppressTiming();
    }

    _expected_samples -= b.getInterval();

    selfmerge();
    //samplesDesc().print("SinkSource received samples");
    //_expected_samples.print("SinkSource expected samples");
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
        QMutexLocker l(&_cache_mutex);

        BOOST_FOREACH( const pBuffer& s, _cache) {
            if (s->sample_offset <= firstSample && s->sample_offset + s->number_of_samples() > firstSample )
            {
                if(D) TaskTimer("%s: sinksource [%u, %u] got [%u, %u]",
                             __FUNCTION__,
                             firstSample,
                             firstSample+numberOfSamples,
                             s->getInterval().first,
                             s->getInterval().last).suppressTiming();
                return s;
            }
        }
    }

    if (_cache.empty())
		return pBuffer();

    TaskTimer(TaskTimer::LogVerbose, "SILENT!").suppressTiming();
    pBuffer b( new Buffer(firstSample, numberOfSamples, sample_rate()));
    memset( b->waveform_data->getCpuMemory(), 0, b->waveform_data->getSizeInBytes1D() );
    return b;
}

unsigned SinkSource::
        sample_rate()
{
    QMutexLocker l(&_cache_mutex);

    if (_cache.empty())
        return (unsigned)-1;
    return _cache.front()->sample_rate;
}

long unsigned SinkSource::
        number_of_samples()
{
    unsigned n = 0;

    QMutexLocker l(&_cache_mutex);

    BOOST_FOREACH( const pBuffer& s, _cache) {
        n += s->number_of_samples();
    }

    return n;
}

pBuffer SinkSource::
        first_buffer()
{
    QMutexLocker l(&_cache_mutex);
    if (_cache.empty())
        return pBuffer();
    return _cache.front();
}

bool SinkSource::empty()
{
    QMutexLocker l(&_cache_mutex);

    return _cache.empty();
}

unsigned SinkSource::size()
{
    QMutexLocker l(&_cache_mutex);
    return _cache.size();
}

SamplesIntervalDescriptor SinkSource::
        samplesDesc()
{
    QMutexLocker l(&_cache_mutex);

    SamplesIntervalDescriptor sid;

    BOOST_FOREACH( const pBuffer& s, _cache) {
        sid |= s->getInterval();
    }

    return sid;
}

} // namespace Signal
