#include "sinksource.h"

#ifndef SAWE_NO_SINKSOURCE_MUTEX
#include <QMutexLocker>
#endif
#include <sstream>
#include <neat_math.h>

#include <boost/foreach.hpp>


//#define TIME_SINKSOURCE
#define TIME_SINKSOURCE if(0)


using namespace std;

namespace Signal {


SinkSource::
        SinkSource( int num_channels )
    :   _num_channels( num_channels )
{
}


SinkSource::
        SinkSource( const SinkSource& b)
            :
        Sink(b),
        _cache( b._cache ),
        _invalid_samples( b._invalid_samples ),
        _num_channels( b._num_channels )
{
}


SinkSource& SinkSource::
        operator=( const SinkSource& b)
{
    _cache = b._cache;
    _invalid_samples = b._invalid_samples;
    _num_channels = b._num_channels;
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
        allocateCache( Signal::Interval I, float fs )
{
#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif

    const int chunkSize = 1<<22; // 16 MB = sizeof(float)*(1<<22)
    I.first = align_down(I.first, chunkSize);
    I.last = align_up(I.last, chunkSize);

    for (std::vector<pBuffer>::iterator itr = findBuffer( I.first );
         itr != _cache.end() || I; itr++)
    {
        if (itr != _cache.end())
        {
            Interval J = (*itr)->getInterval();
            if (J & Interval(I.first, I.first+1))
            {
                I.first = J.last;
                continue;
            }
        }

        pBuffer n( new Buffer( I.first, chunkSize, fs, _num_channels) );
        itr = _cache.insert(itr, n);
        I.first += chunkSize;
    }
}

/*
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

    const int chunkSize = 1<<22; // 16 MB = sizeof(float)*(1<<22)
    //const int chunkSize = 0; // Chunks are disabled, Buffer:s are cached as they are computed
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
}*/


void SinkSource::
        merge( pBuffer bp )
{
    TIME_SINKSOURCE
            TaskTimer tt("%s %s & %s = %s. %d buffers", __FUNCTION__,
                         bp->getInterval().toString().c_str(),
                         _valid_samples.toString().c_str(),
                         (bp->getInterval() & _valid_samples).toString().c_str(),
                         _cache.size());

    bp->release_extra_resources();

    const Buffer& b = *bp;

    allocateCache(b.getInterval(), b.sample_rate());

#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker cache_locker(&_cache_mutex);
#endif

    if (!_cache.empty())
        EXCEPTION_ASSERT(_cache.front()->sample_rate() == b.sample_rate());

    for( std::vector<pBuffer>::iterator itr = findBuffer(b.getInterval().first); itr!=_cache.end(); itr++ )
        **itr |= b;

    _invalid_samples -= b.getInterval();
    _valid_samples |= b.getInterval();
}


void SinkSource::
        invalidate_samples(const Intervals& I)
{
    _invalid_samples |= I;
    _valid_samples -= I;
}

void SinkSource::
        invalidate_and_forget_samples(const Intervals& I)
{
    invalidate_samples( I );
    // could discard allocated data here, never mind...
    //selfmerge( I );
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
    TIME_SINKSOURCE TaskTimer tt("%s %s from %s", __FUNCTION__, I.toString().c_str(), _valid_samples.toString().c_str());

    Interval validFetch = (I & _valid_samples).fetchFirstInterval();

    if (!validFetch)
        validFetch.first = I.last;
    if (validFetch.first > I.first)
    {
        EXCEPTION_ASSERT( Interval(I.first, validFetch.first) );
        return zeros(Interval(I.first, validFetch.first));
    }

    std::vector<pBuffer>::const_iterator itr = findBuffer(I.first);

    EXCEPTION_ASSERT( itr != _cache.end() );

    pBuffer b = *itr;
    if (!(b->getInterval () & Interval(I.first, I.first+1)))
    {
        TaskInfo("!(b->getInterval () & Interval(I.first, I.first+1))");
        TaskInfo("_valid_samples = %s", _valid_samples.toString ().c_str ());
        TaskInfo("I = %s", I.toString ().c_str ());
        TaskInfo("b->getInterval() = %s", b->getInterval().toString ().c_str ());

        EXCEPTION_ASSERT( _valid_samples & Interval(I.first, I.first+1) );
    }

    if ((b->getInterval() & _valid_samples) == b->getInterval())
    {
        EXCEPTION_ASSERT( Interval(I.first, I.first+1) & b->getInterval() );
        return b;
    }

    validFetch &= b->getInterval();

    if (!validFetch)
    {
        TaskInfo("I = %s", I.toString ().c_str ());
        TaskInfo("_valid_samples = %s", _valid_samples.toString ().c_str ());
        TaskInfo("b->getInterval() = %s", b->getInterval().toString ().c_str ());
        TaskInfo("I & _valid_samples = %s", (I & _valid_samples).toString ().c_str ());
        TaskInfo("(I & _valid_samples).fetchFirstInterval() = %s", (I & _valid_samples).fetchFirstInterval().toString().c_str());
        TaskInfo("(I & _valid_samples).fetchFirstInterval() & b->getInterval() = %s", ((I & _valid_samples).fetchFirstInterval() & b->getInterval()).toString().c_str());
        Interval validFetch = (I & _valid_samples).fetchFirstInterval();
        TaskInfo("validFetch = %s", validFetch.toString ().c_str ());
        validFetch &= b->getInterval();
        TaskInfo("validFetch = %s", validFetch.toString ().c_str ());
        TaskInfo ti("_cache[%d] = ", _cache.size ());
        BOOST_FOREACH( pBuffer b, _cache )
            TaskInfo("b->getInterval() = %s", b->getInterval ().toString ().c_str ());

        EXCEPTION_ASSERT( validFetch );
        EXCEPTION_ASSERT( false );
    }

    // TODO: if COW chunks could be created with an offset we could return all of b that is valid instead of just a copy of the portion that matches I. Less copying, returning more data right away.
    pBuffer n(new Buffer(validFetch, b->sample_rate(), b->number_of_channels ()));
    *n |= *b;

    return n;
}


float SinkSource::
        sample_rate()
{
#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif

    if (_cache.empty())
        return 44100;

    return _cache.front()->sample_rate();
}


IntervalType SinkSource::
        number_of_samples()
{
    return _valid_samples.spannedInterval().count();
}


Interval SinkSource::
        getInterval()
{
    return _valid_samples.spannedInterval();
}


pBuffer SinkSource::
        first_buffer()
{
#ifndef SAWE_NO_SINKSOURCE_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif
    EXCEPTION_ASSERT( !_cache.empty() );

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

/*
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
*/

bool cache_search( IntervalType t, const pBuffer& b )
{
    return t < b->getInterval().last;
}


std::vector<pBuffer>::iterator SinkSource::
        findBuffer( Signal::IntervalType sample )
{
    return upper_bound(_cache.begin(), _cache.end(), sample, cache_search);
}

} // namespace Signal
