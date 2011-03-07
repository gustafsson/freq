#include "signal/operationcache.h"

static const bool D = false;

namespace Signal {

OperationCache::
        OperationCache( pOperation source )
:   Operation(source),
    _cache()
{
    _cache.setNumChannels(source->num_channels());
}


pBuffer OperationCache::
        read( const Interval& I )
{
    static bool enable_cache = true;

    Interval ok = (~affected_samples() & I).fetchFirstInterval();

    // TODO do this
    /*if (ok.first == I.first && ok.count() && enable_cache)
    {
        // Wouldn't affect these samples, don't bother caching the results
        pBuffer b = source()->readFixedLength( ok );
        if (D) TaskTimer("%s: cache not affecting [%u, %u] got [%u, %u]",
                     __FUNCTION__,
                     I.first,
                     I.last,
                     b->getInterval().first,
                     b->getInterval().last).suppressTiming();
        return b;
    }*/


    // cached samples doesn't count in samplesDesc if they are marked as invalid
    Intervals cached = _cache.samplesDesc_current_channel() - _cache.invalid_samples_current_channel();

    ok = (cached & I).fetchFirstInterval();

    if (ok.first == I.first && ok.count() && enable_cache)
    {
        // Don't need anything new, return cache
        pBuffer b = _cache.readFixedLength( ok );
        if (D) TaskTimer("%s: cache [%u, %u] got [%u, %u]",
                     __FUNCTION__,
                     I.first,
                     I.last,
                     b->getInterval().first,
                     b->getInterval().last).suppressTiming();
        return b;
    }

    Interval missing = Intervals(I) - cached;

    pBuffer b = readRaw( missing );

    if (D) TaskTimer tt("%s: raw [%u, %u] got [%u, %u]",
                 __FUNCTION__,
                 I.first,
                 I.last,
                 b?b->getInterval().first:0,
                 b?b->getInterval().last:0);

    unsigned c = get_channel();

    if (b)
    {
        Signal::Interval J = b->getInterval();

        _cache.put(b);

        if (_invalid_returns[c] & J)
        {
            Operation::invalidate_samples( _invalid_returns[c] & J );
            _invalid_returns[c] -= J;
        }

        cached = _cache.samplesDesc_current_channel() - _cache.invalid_samples_current_channel();
        ok = (cached & I).fetchFirstInterval();
        if (ok.first == I.first && ok.count())
        {
            return _cache.readFixedLength( ok );
        }

        missing = (Intervals(I) - cached).fetchFirstInterval();
    }

    _invalid_returns[c] |= missing;
    b = source()->readFixedLength( missing );
    _cache.put(b);
    return b;
}


void OperationCache::
        invalidate_samples(const Intervals& I)
{
    _invalid_returns.resize( num_channels() );
    _cache.setNumChannels( num_channels() );
    _cache.invalidate_samples( I );
    // TODO do this
    //_cache.invalidate_and_forget_samples(I);
    Operation::invalidate_samples( I );
}


Intervals OperationCache::
        invalid_samples()
{
    Intervals c = _cache.invalid_samples_all_channels();
    return c;
}


Intervals OperationCache::
        cached_samples()
{
    return _cache.samplesDesc_all_channels() - _cache.invalid_samples_all_channels() - invalid_returns();
}


Intervals OperationCache::
        invalid_returns()
{
    Intervals R;
    for (unsigned i=0; i<_invalid_returns.size(); ++i)
        R |= _invalid_returns[i];

    return R;
}


unsigned OperationCache::
        num_channels()
{
    return _cache.num_channels();
}


void OperationCache::
        set_channel(unsigned c)
{
    _cache.set_channel( c );
    Operation::set_channel( c );
}


unsigned OperationCache::
        get_channel()
{
    return _cache.get_channel();
}


void OperationCache::
        source(pOperation v)
{
    _cache.setNumChannels(num_channels());
    Operation::source( v );
}


} // namespace Signal
