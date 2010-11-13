#include "sinksource.h"

#include <boost/foreach.hpp>
#include <QMutexLocker>
#include <sstream>

static const bool D = false;

using namespace std;

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

    merge(b);
}


void SinkSource::
        putExpectedSamples( pBuffer buffer, const Intervals& expected )
{
    BufferSource bs( buffer );

    Intervals I = expected & buffer->getInterval();
    BOOST_FOREACH( const Interval& i, I )
    {
        pBuffer s = bs.readFixedLength( i );
        put( s );
    }
}


// todo remove
static bool bufferLessThan(const pBuffer& a, const pBuffer& b)
{
    return (IntervalType)a->sample_offset < (IntervalType)b->sample_offset;
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

    Intervals sid = samplesDesc();
	std::vector<pBuffer> new_cache;

    BOOST_FOREACH( Interval i, sid )
	{
        for (unsigned L=0; i.first < i.last; i.first+=L)
		{
            L = lpo2s( i.count() + 1);

            if (L < (1<<13))
                L = i.count();

            new_cache.push_back(readFixedLength( Interval( i.first, i.first + L) ));
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
    bp->release_extra_resources();

    const Buffer& b = *bp;
    float FS = sample_rate();

    QMutexLocker cache_locker(&_cache_mutex);

    if (!_cache.empty())
        BOOST_ASSERT(_cache.front()->sample_rate == b.sample_rate);

    std::stringstream ss;

    // REMOVE caches that become outdated by this new buffer 'b'
    for ( std::vector<pBuffer>::iterator itr = _cache.begin(); itr!=_cache.end(); )
    {
        const pBuffer ps = *itr;
        const Buffer& s = *ps;

        Intervals toKeep = s.getInterval();
        toKeep -= b.getInterval();

        Intervals toRemove = s.getInterval();
        toRemove &= b.getInterval();

        if (toRemove)
        {
            if(D) ss << " -" << s.getInterval().toString();

            itr = _cache.erase(itr); // Note: 'pBuffer s' stores a copy for the scope of the for-loop

            BOOST_FOREACH( Interval i, toKeep )
            {
                if(D) ss << " +" << i.toString();

                pBuffer n( new Buffer( i.first, i.count(), FS));
                GpuCpuData<float>* dest = n->waveform_data();
                memcpy( dest->getCpuMemory(),
                        s.waveform_data()->getCpuMemory() + (i.first - (IntervalType)s.sample_offset),
                        dest->getSizeInBytes1D() );
                itr = _cache.insert(itr, n );
                itr++; // Move past inserted element
            }
        } else {
            itr++;
        }
    }

    pBuffer n( new Buffer( b.sample_offset, b.number_of_samples(), b.sample_rate));
    GpuCpuData<float>* src = b.waveform_data();
    memcpy( n->waveform_data()->getCpuMemory(),
            src->getCpuMemory(),
            src->getSizeInBytes1D());
    _cache.push_back( n );
    cache_locker.unlock(); // done with _cache, samplesDesc() below needs the lock

    if(D) if (!ss.str().empty())
    {
        ss << " +" << n->getInterval().toString();
        TaskTimer("M:%s", ss.str().c_str()).suppressTiming();
    }

    _invalid_samples -= b.getInterval();

    selfmerge();
    //samplesDesc().print("SinkSource received samples");
    //_expected_samples.print("SinkSource expected samples");
}


void SinkSource::
        reset()
{
    QMutexLocker l(&_cache_mutex);
    _cache.clear();
    _invalid_samples = Intervals();
}


pBuffer SinkSource::
        read( const Interval& I )
{
    {
        QMutexLocker l(&_cache_mutex);

        BOOST_FOREACH( const pBuffer& s, _cache) {
            if (s->sample_offset <= I.first && s->sample_offset + s->number_of_samples() > I.first )
            {
                if(D) TaskTimer("%s: sinksource [%u, %u] got [%u, %u]",
                             __FUNCTION__,
                             I.first,
                             I.last,
                             s->getInterval().first,
                             s->getInterval().last).suppressTiming();
                return s;
            }
        }
    }

    TaskTimer(TaskTimer::LogVerbose, "SILENT!").suppressTiming();
    return zeros(I);
}


float SinkSource::
        sample_rate()
{
    QMutexLocker l(&_cache_mutex);

    if (_cache.empty())
        return 0;

    return _cache.front()->sample_rate;
}


long unsigned SinkSource::
        number_of_samples()
{
    return samplesDesc().coveredInterval().count();
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


Intervals SinkSource::
        samplesDesc()
{
    QMutexLocker l(&_cache_mutex);

    Intervals sid;

    BOOST_FOREACH( const pBuffer& s, _cache) {
        sid |= s->getInterval();
    }

    return sid;
}

} // namespace Signal
