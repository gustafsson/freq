#include "sinksource.h"

#ifndef SAWE_NO_SINKSOURCE_MUTEX
#include <QMutexLocker>
#endif
#include <sstream>
#include <neat_math.h>

#include <boost/foreach.hpp>


// Set to 'true' to enable debugging output
static const bool D = false;

//#define TIME_SINKSOURCE
#define TIME_SINKSOURCE if(0)


using namespace std;

namespace Signal {


bool cache_search( IntervalType t, const pBuffer& b )
{
    return t < b->getInterval().last;
}


SinkSource::
        SinkSource()
{
}


SinkSource::
        SinkSource( const SinkSource& b)
            :
        Sink(b),
        _cache( b._cache ),
        _invalid_samples( b._invalid_samples )
{
}


SinkSource& SinkSource::
        operator=( const SinkSource& b)
{
    _cache = b._cache;
    _invalid_samples = b._invalid_samples;
    return *this;
}


void SinkSource::
        put( pBuffer b )
{
    TIME_SINKSOURCE TaskTimer tt("%s %s", __FUNCTION__, b->getInterval().toString().c_str());

    /* CANONICAL {
        QMutexLocker l(&_mutex);

        // Simply remove previous overlapping buffer, don't bother merging.
        foreach( pBuffer& s, _cache) {
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

    const Intervals I = expected & buffer->getInterval();
    BOOST_FOREACH( const Interval& i, I )
    {
        pBuffer s = bs.readFixedLength( i );
        put( s );
    }
}


void SinkSource::
        selfmerge( Signal::Intervals forget )
{
    TIME_SINKSOURCE TaskTimer tt("%s %s %d", __FUNCTION__, forget.toString().c_str(), _cache.size());

    {
#ifndef SAWE_NO_SINKSOURCE_MUTEX
        QMutexLocker l(&_cache_mutex);
#endif
        if (_cache.empty())
            return;
    }

    // const int chunkSize = 1<<22; // 16 MB = sizeof(float)*(1<<22)
    const int chunkSize = 0; // Chunks are disabled, Buffer:s are cached as they are computed
    if (0==chunkSize && forget.empty())
        return;

    const Intervals sid = samplesDesc() - forget;
    std::vector<pBuffer> new_cache;
    new_cache.reserve(_cache.size());

    BOOST_FOREACH( const Interval& i, sid )
    {
        IntervalType j=i.first;
        while (j < i.last)
        {
            Interval J(j,j+chunkSize);
            if (0 == chunkSize)
                J.last = i.last;
            else
                J.last = min(i.last, align_down(J.last, chunkSize));

            pBuffer b;
            if (J.count() == chunkSize)
            {
                b = readFixedLength( J );
            }
            else
            {
                b = read(J);
                // assume b was a valid read (i.e that J.first is spanned by b->getInterval())
                if ((b->getInterval() & J) != b->getInterval())
                {
                    // If b is smaller than or equal to J, that's ok as long as it spans J.first (which is should by contract).
                    // If b is bigger than J it needs to be shrinked. Let readFixedLength do the job.
                    b = BufferSource(b).readFixedLength(J & b->getInterval());
                }
            }
            new_cache.push_back( b );
            j = b->getInterval().last;
        }
    }

    {
#ifndef SAWE_NO_SINKSOURCE_MUTEX
        QMutexLocker l(&_cache_mutex);
#endif
        _cache = new_cache;
    }
}


void SinkSource::
        merge( pBuffer bp )
{
    TIME_SINKSOURCE
            TaskTimer tt("%s %s & %s = %s. %d buffers", __FUNCTION__,
                         bp->getInterval().toString().c_str(),
                         samplesDesc().toString().c_str(),
                         (bp->getInterval() & samplesDesc()).toString().c_str(),
                         _cache.size());

    bp->release_extra_resources();

    const Buffer& b = *bp;
    float FS = sample_rate();

    std::stringstream ss;
    pBuffer n;

	{
#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker cache_locker(&_cache_mutex);
#endif

    if (!_cache.empty())
        BOOST_ASSERT(_cache.front()->sample_rate == b.sample_rate);

    // REMOVE caches that become outdated by this new buffer 'b'
    std::vector<pBuffer>::iterator itr = upper_bound(_cache.begin(), _cache.end(), b.getInterval().first, cache_search);

    while( itr!=_cache.end() )
    {
        pBuffer s = *itr;

        if (s->getInterval().first >= b.getInterval().last )
            break;

        Interval toRemove = s->getInterval();
        toRemove &= b.getInterval();

        if (toRemove)
        {
            TIME_SINKSOURCE TaskTimer tt("Removing %s from %s", toRemove.toString().c_str(), s->getInterval().toString().c_str());
            if(D) ss << " -" << s->getInterval().toString();

            // expensive, should possibly change type to list instead of vector.
            itr = _cache.erase(itr); // Note: 'pBuffer s' stores a copy for the scope of the for-loop

            Intervals toKeep = s->getInterval();
            toKeep -= b.getInterval();

            BOOST_FOREACH( const Interval& i, toKeep )
            {
                if(D) ss << " +" << i.toString();

                pBuffer n( new Buffer( i.first, i.count(), FS));
                *n |= *s;
                itr = _cache.insert(itr, n );
                itr++; // Move past inserted element
            }

            if (itr != _cache.begin())
                itr--;
        } else {
            itr++;
        }
    }

    n = pBuffer( new Buffer( b.sample_offset, b.number_of_samples(), b.sample_rate));
    *n |= b;
    itr = _cache.insert(itr, bp );
    } //cache_locker.unlock(); // finished working with _cache, samplesDesc() below needs the lock

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
        invalidate_and_forget_samples(const Intervals& I)
{
    invalidate_samples( I );
    selfmerge( I );
}


void SinkSource::
        clear()
{
#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif
    _cache.clear();
    _invalid_samples = Intervals();
}


pBuffer SinkSource::
        read( const Interval& I )
{
    TIME_SINKSOURCE TaskInfo tt("%s %s from %s", __FUNCTION__, I.toString().c_str(), samplesDesc().toString().c_str());

#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif
    if(D) BOOST_FOREACH( const pBuffer& b, _cache )
    {
        TaskInfo("%s", b->getInterval().toString().c_str());
    }

    std::vector<pBuffer>::const_iterator itr = upper_bound(_cache.begin(), _cache.end(), I.first, cache_search);

    Interval not_found = I;
    if (itr != _cache.end())
    {
        Interval j = (**itr).getInterval();
        if (Interval(I.first, I.first+1) & j)
        {
            if(D) TaskInfo("Found %d in %s", (int)I.first, j.toString().c_str());

            return *itr;
        }
        if (j.first < not_found.last)
            not_found.last = j.first;
    }

#ifndef SAWE_NO_SINKSOURCE_MUTEX
    l.unlock();
#endif

    BOOST_ASSERT( (samplesDesc() & not_found).empty() );
    if(D) TaskInfo("zeros %s", not_found.toString().c_str());

    // Return silence
    return zeros(not_found);
}


float SinkSource::
        sample_rate()
{
#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif

    if (_cache.empty())
        return 44100;

    return _cache.front()->sample_rate;
}


IntervalType SinkSource::
        number_of_samples()
{
    return samplesDesc().spannedInterval().count();
}


Interval SinkSource::
        getInterval()
{
    return samplesDesc().spannedInterval();
}


pBuffer SinkSource::
        first_buffer()
{
#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif
    BOOST_ASSERT( !_cache.empty() );

    if (_cache.empty())
        return pBuffer();
    return _cache.front();
}


bool SinkSource::empty()
{
#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif
    return _cache.empty();
}


Intervals SinkSource::
        samplesDesc()
{
#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif

    Intervals sid;

    BOOST_FOREACH( const pBuffer& s, _cache) {
        sid |= s->getInterval();
    }

    return sid;
}

} // namespace Signal
