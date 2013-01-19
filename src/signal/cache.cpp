#include "cache.h"

#include "TaskTimer.h"
#include "exceptionassert.h"
#include "neat_math.h"

namespace Signal {

Cache::
        Cache( )
{
}


Cache::
        Cache( const Cache& b)
{
    *this = b;
}


Cache& Cache::
        operator=( const Cache& b)
{
    _cache = b._cache;
    return *this;
}


void Cache::
        put( pBuffer bp )
{
    bp->release_extra_resources();

    const Buffer& b = *bp;

    allocateCache(b.getInterval(), b.sample_rate(), b.number_of_channels ());

    for( std::vector<pBuffer>::iterator itr = findBuffer(b.getInterval().first); itr!=_cache.end(); itr++ )
        **itr |= b;

    _valid_samples |= b.getInterval();
}


void Cache::
        allocateCache( Signal::Interval I, float fs, int num_channels )
{
    int N = this->num_channels ();
    if (N>0 && N != num_channels)
        throw std::invalid_argument(str(boost::format("Expected %d channels, got %d\n%s") %
                                        N % num_channels % BACKTRACE()));

    float F = this->sample_rate ();
    if (F>0 && F != fs) // Not fuzzy compare, must be identical.
        throw std::invalid_argument(str(boost::format("Expected fs=%g channels, got %g\n%s") %
                                        F % fs % BACKTRACE()));

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

        pBuffer n( new Buffer( I.first, chunkSize, fs, num_channels) );
        itr = _cache.insert(itr, n);
        I.first += chunkSize;
    }
}


void Cache::
        invalidate_samples(const Intervals& I)
{
    _valid_samples -= I;
}


void Cache::
        clear()
{
    _cache.clear();
    _valid_samples = Intervals();
}


pBuffer Cache::
        read( const Interval& I ) const
{
    // COPY FROM SourceBase::readFixedLength

    // Try a simple read
    pBuffer p = readAtLeastFirstSample( I );
    if (I == p->getInterval())
        return p;

    // Didn't get exact result, prepare new Buffer
    pBuffer r( new Buffer(I, p->sample_rate(), p->number_of_channels ()) );

    for (unsigned c=0; c<r->number_of_channels (); ++c)
    {
        // Allocate cpu memory and prevent calling an unnecessary clear by flagging the store as up-to-date
        // TODO make allocation in same memory as p was allocated in.
        // CpuMemoryStorage::WriteAll<3>( r->getChannel (c)->waveform_data() );

//        if (p->getChannel (c)->waveform_data()->HasValidContent<CudaGlobalStorage>())
//            CudaGlobalStorage::WriteAll<3>( r->getChannel (c)->waveform_data() );
//        else
//            CpuMemoryStorage::WriteAll<3>( r->getChannel (c)->waveform_data() );
    }

    Intervals sid(I);

    do
    {
        if (!p)
            p = readAtLeastFirstSample( sid.fetchFirstInterval() );

        sid -= p->getInterval();
        (*r) |= *p; // Fill buffer
        p.reset();
    } while (sid);

    return r;
}


pBuffer Cache::
        readAtLeastFirstSample( const Interval& I ) const
{
    EXCEPTION_ASSERT_LESS( I.first, I.last );

    Interval validFetch = (I & _valid_samples).fetchFirstInterval();

    if (!validFetch)
    {
        // Assume I.last > I.first and return zeros of Interval(I.first, validFetch.first).
        validFetch.first = I.last;
    }

    if (validFetch.first > I.first)
    {
        EXCEPTION_ASSERT( Interval(I.first, validFetch.first) );
        pBuffer zeros = pBuffer( new Buffer(Interval(I.first, validFetch.first), sample_rate(), num_channels()) );
        return zeros;
    }

    std::vector<pBuffer>::const_iterator itr = findBuffer(I.first);

    EXCEPTION_ASSERT( itr != _cache.end() );

    pBuffer b = *itr;
    EXCEPTION_ASSERT( b->getInterval().contains (I.first) );
    if ((b->getInterval() & _valid_samples) == b->getInterval())
        return b;

    validFetch &= b->getInterval();

    // TODO: if COW chunks could be created with an offset we could return all
    // of b that is valid instead of just a copy of the portion that matches I.
    // Would result in less copying by returning more data right away.

    pBuffer n(new Buffer(validFetch, b->sample_rate(), b->number_of_channels ()));
    *n |= *b;

    return n;
}


float Cache::
        sample_rate() const
{
    if (_cache.empty())
        return -1;

    return _cache.front()->sample_rate();
}


int Cache::
        num_channels() const
{
    if (_cache.empty())
        return -1;

    return _cache.front()->number_of_channels();
}


Intervals Cache::
        samplesDesc() const
{
    Intervals r = _valid_samples;
    return r;
}


class cache_search
{
public:
    bool operator()( IntervalType t, const pBuffer& b )
    {
        return t < b->getInterval().last;
    }

#if defined(_MSC_VER) && defined(_DEBUG)
    // Integrity checks in windows debug mode
    bool operator()( const pBuffer& b, IntervalType t )
    {
        return b->getInterval().last < t;
    }

    bool operator()( const pBuffer& b, const pBuffer& b2 )
    {
        return b->getInterval().last < b2->getInterval().last;
    }
#endif
};


std::vector<pBuffer>::iterator Cache::
        findBuffer( Signal::IntervalType sample )
{
    return upper_bound(_cache.begin(), _cache.end(), sample, cache_search());
}

std::vector<pBuffer>::const_iterator Cache::
        findBuffer( Signal::IntervalType sample ) const
{
    return upper_bound(_cache.begin(), _cache.end(), sample, cache_search());
}

} // namespace Signal

namespace Signal {

void Cache::
        test()
{
    Cache cache;

    pBuffer b(new Buffer(Interval(1, 17), 5.1, 7));
    EXCEPTION_ASSERT (cache.samplesDesc ().empty ());
    cache.put (b);
    EXCEPTION_ASSERT_EQUALS (cache.samplesDesc (), b->getInterval () );
    EXCEPTION_ASSERT_EQUALS (cache.sample_rate (), b->sample_rate () );
    EXCEPTION_ASSERT_EQUALS (cache.num_channels (), (int)b->number_of_channels () );

    cache.put (pBuffer(new Buffer(Interval(18, 75), 5.1, 7)));
    EXCEPTION_ASSERT_EQUALS (cache.samplesDesc (), Intervals(b->getInterval ()) | Intervals(18, 75) );

    cache.put (pBuffer(new Buffer(Interval(14, 25), 5.1, 7)));
    EXCEPTION_ASSERT_EQUALS (cache.samplesDesc (), Interval(1, 75) );

    for (int c=0; c<(int)b->number_of_channels (); ++c)
    {
        pMonoBuffer mono = b->getChannel (c);
        float *p = mono->waveform_data ()->getCpuMemory ();
        for (int i=0; i<b->number_of_samples (); ++i)
            p[i] = c;
    }
    cache.put (b);
    pBuffer r = cache.read (Interval(-1, 80));
    EXCEPTION_ASSERT_EQUALS (r->getInterval (), Interval(-1, 80) );
    EXCEPTION_ASSERT_EQUALS (r->number_of_channels (), b->number_of_channels ());
    for (int c=0; c<(int)r->number_of_channels (); ++c)
    {
        pMonoBuffer mono = r->getChannel (c);
        float *p = mono->waveform_data ()->getCpuMemory ();
        for (int i=0; i<r->number_of_samples (); ++i)
        {
            IntervalType j = r->getInterval ().first + i;
            EXCEPTION_ASSERT_EQUALS (p[i], b->getInterval ().contains ( j ) ? c : 0 );
        }
    }

    try {
        cache.put (pBuffer(new Buffer(Interval(14, 25), 5.11, 7)));
        EXCEPTION_ASSERTX( false, "expected an exception to be thrown when supplying a non-consistent sample rate" );
    } catch (const std::invalid_argument&) {}

    try {
        cache.put (pBuffer(new Buffer(Interval(14, 25), 5.1, 6)));
        EXCEPTION_ASSERTX( false, "expected an exception to be thrown when supplying a non-consistent number of channels" );
    } catch (const std::invalid_argument&) {}
}

} // namespace Signal
