#include "cache.h"

#include "tasktimer.h"
#include "log.h"
#include "exceptionassert.h"
#include "neat_math.h"
#include "backtrace.h"
#include "timer.h"
#include "cpumemorystorage.h"

using namespace boost;

namespace Signal {

static Buffer zeros(Signal::Interval I, float fs, int C)
{
    Buffer b(I,fs,C);

    // DataStorage clears data on read access of uninitialized data
    for (int i=0; i<C; i++)
        CpuMemoryStorage::ReadOnly<1>(b.getChannel (i)->waveform_data ());

    return b;
}


static pBuffer zerosp(Signal::Interval I, float fs, int C)
{
    return pBuffer(new Buffer(zeros(I, fs, C)));
}


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

    Timer t;
    for( std::vector<pBuffer>::iterator itr = findBuffer(b.getInterval().first); itr!=_cache.end(); itr++ )
    {
        Buffer& t = **itr;
        if (t.getInterval ().first >= b.getInterval ().last)
            break;
        t |= b;
    }

    _valid_samples |= b.getInterval();

    double T=t.elapsed ();
    if (T>10e-3)
        TaskInfo(boost::format("!!! It took %s to put(%s) into cache") % TaskTimer::timeToString(T) % b.getInterval ());
}


void Cache::
        allocateCache( const Interval& J, float fs, int num_channels )
{
    Interval I = J;
    int N = this->num_channels ();
    float F = this->sample_rate ();
    if (empty()) {
        N = num_channels;
        F = fs;
    } else {
        if (N != num_channels)
            BOOST_THROW_EXCEPTION(InvalidBufferDimensions() << errinfo_format
                                  (boost::format("Expected %d channels, got %d") %
                                            N % num_channels) << Backtrace::make ());

        if (F != fs) // Not fuzzy compare, must be identical.
            BOOST_THROW_EXCEPTION(InvalidBufferDimensions() << errinfo_format
                                  (boost::format("Expected fs=%g, got %g") %
                                   F % fs) << Backtrace::make ());
    }

    // chunkSize 1 << ...
    // 22 -> 1.8 ms
    // 21 -> 1.5 ms
    // 20 -> 1.2 ms -> 4 MB cache chunks
    // 19 -> 1.2 ms
    // 18 -> 1.2 ms
    // 10 -> 1.2 ms
    const IntervalType chunkSize = 1<<20;
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

        //TaskTimer tt(boost::format("Allocating cache %s") % Interval(I.first, I.first+chunkSize));

        pBuffer n;
        if (!_discarded.empty ())
        {
            n = _discarded.back ();
            _discarded.pop_back ();
            n->set_sample_offset (I.first);
            n->set_sample_rate (fs);
        }

        if (!n || n->number_of_channels () != unsigned(num_channels) || n->number_of_samples () != chunkSize)
            n.reset ( new Buffer( I.first, chunkSize, fs, num_channels) );

        for (int i=0; i<num_channels; i++)
        {
            // Force allocation here, don't delay it.
            //
            // Don't bother clearing the buffer with zeros. _valid_samples keeps
            // track of what data we can readily use.
            //CpuMemoryStorage::WriteAll<1>(n->getChannel (i)->waveform_data ());
        }

        itr = _cache.insert(itr, n);
        I.first += chunkSize;
    }
}


void Cache::
        invalidate_samples(const Intervals& I)
{
    _valid_samples -= I;
}


Signal::Intervals Cache::
        purge(Signal::Intervals still_needed)
{
    Signal::Intervals purged;

    for (auto itr = _cache.begin (); itr != _cache.end ();)
    {
        pBuffer b = *itr;
        Signal::Interval i = b->getInterval();
        if (i & still_needed)
        {
            itr++;
            continue;
        }
        else
        {
            invalidate_samples (i);
            purged |= i;
            itr = _cache.erase (itr);

            if (!b->getChannel (0)->waveform_data ()->HasValidContent<CpuMemoryStorage>())
                Log("purging non-allocated buffer") % i;

            if (_discarded.size ()<2)
                _discarded.push_back (b);
        }
    }

    return purged;
}


size_t Cache::
        cache_size() const
{
    size_t sz = 0;

    for (pBuffer const& b : _cache)
        for (unsigned c=0; c<b->number_of_channels (); c++)
            if (b->getChannel (c)->waveform_data ()->HasValidContent<CpuMemoryStorage>())
                sz += b->getChannel (c)->number_of_samples ();

    return sz * sizeof(Signal::TimeSeriesData::element_type);
}


void Cache::
        clear()
{
    _cache.clear ();
    _discarded.clear ();
    _valid_samples = Intervals();
}


pBuffer Cache::
        read( const Interval& I ) const
{
    pBuffer r = pBuffer( new Buffer(I, sample_rate(), num_channels ()) );
    read(r);
    return r;
}


void Cache::
        read( pBuffer r ) const
{
    Timer t;

    Intervals sid(r->getInterval ());
    while (sid)
    {
        Interval s = sid.fetchFirstInterval ();
        pBuffer p = readAtLeastFirstSample( s );
        *r |= *p; // Fill buffer
        sid -= p->getInterval();
    }

    // set remaining samples to zero
    for (Interval i : sid)
        *r |= zeros(i, sample_rate(), num_channels ());

    double T=t.elapsed ();
    if (T > 10e-3 && T/r->number_of_samples ()>10e-6)
        TaskInfo(boost::format("!!! It took %s to read(%s) from cache") % TaskTimer::timeToString(T) % r->number_of_samples ());
}


pBuffer Cache::
        readAtLeastFirstSample( const Interval& I ) const
{
    EXCEPTION_ASSERT_LESS_DBG( I.first, I.last );

    Interval validFetch = (I & _valid_samples).fetchFirstInterval();

    if (!validFetch)
        return zerosp(I, sample_rate(), num_channels ());

    if (validFetch.first > I.first)
        return zerosp(Interval(I.first, validFetch.first), sample_rate(), num_channels());

    // Find the cache chunk for sample I.first
    std::vector<pBuffer>::const_iterator itr = findBuffer(I.first);

    EXCEPTION_ASSERT_DBG( itr != _cache.end() );

    pBuffer b = *itr;
    Interval bI = b->getInterval ();
    EXCEPTION_ASSERTX_DBG( bI.contains (I.first), boost::format("bI = %s, I = %s") % bI % I );

    // If all of b is ok, return b
    if ( _valid_samples.contains (bI) )
    {
        // TODO: if COW chunks could be created with an offset we could return all
        // of b that is valid instead of just a copy of the portion that matches I.
        // Would result in less copying by returning more data right away.
        return b;
    }

    validFetch &= bI;

    pBuffer n = pBuffer(new Buffer(validFetch, b->sample_rate(), b->number_of_channels ()));
    *n |= *b;
    return n;
}


float Cache::
        sample_rate() const
{
    if (_cache.empty())
        return 1;

    return _cache.front()->sample_rate();
}


int Cache::
        num_channels() const
{
    if (_cache.empty())
        return 0;

    return _cache.front()->number_of_channels();
}


Intervals Cache::
        samplesDesc() const
{
    return _valid_samples;
}


Interval Cache::
        spannedInterval() const
{
    return _valid_samples.spannedInterval ();
}


Intervals Cache::
        allocated() const
{
    Intervals I;
    for (pBuffer b : _cache)
        I |= b->getInterval ();
    return I;
}


bool Cache::
        contains(const Signal::Intervals& I) const
{
    return _valid_samples.contains (I);
}


bool Cache::
        empty() const
{
    return !_valid_samples;
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
        findBuffer( IntervalType sample )
{
    return upper_bound(_cache.begin(), _cache.end(), sample, cache_search());
}

std::vector<pBuffer>::const_iterator Cache::
        findBuffer( IntervalType sample ) const
{
    return upper_bound(_cache.begin(), _cache.end(), sample, cache_search());
}

} // namespace Signal

#include "test/printbuffer.h"

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
    cache.put (pBuffer(new Buffer(Interval(-3, -1), 5.1, 7)));
    cache.put (pBuffer(new Buffer(Interval(76, 77), 5.1, 7)));
    pBuffer r = cache.read (Interval(-4, 80));

    EXCEPTION_ASSERT_EQUALS (r->getInterval (), Interval(-4, 80) );
    EXCEPTION_ASSERT_EQUALS (r->number_of_channels (), b->number_of_channels ());

    EXCEPTION_ASSERT_EQUALS (cache.samplesDesc (), Intervals(-3, -1) | Interval(1, 75) | Interval(76, 77));

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

    cache.invalidate_samples (Interval(2, 16));
    r = cache.read (Interval(0, 20));
    pBuffer e(new Buffer(Interval(0, 20), 5.1, 7));
    e->mergeChannelData ();
    *e |= *b;
    *e |= Buffer(Interval(2, 16), 5.1, 7);

    if (*r != *e) {
        PRINT_BUFFER(r,"r");
        PRINT_BUFFER(e,"e");
        EXCEPTION_ASSERT( *r == *e );
    }


    try {
        cache.put (pBuffer(new Buffer(Interval(14, 25), 5.11, 7)));
        EXCEPTION_ASSERTX( false, "expected an exception to be thrown when supplying a non-consistent sample rate" );
    } catch (const InvalidBufferDimensions&) {}

    try {
        cache.put (pBuffer(new Buffer(Interval(14, 25), 5.1, 6)));
        EXCEPTION_ASSERTX( false, "expected an exception to be thrown when supplying a non-consistent number of channels" );
    } catch (const InvalidBufferDimensions&) {}
}

} // namespace Signal
