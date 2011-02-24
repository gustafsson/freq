#include "signal/operationcache.h"

static const bool D = false;

namespace Signal {

OperationCache::
        OperationCache( pOperation source )
:   Operation(source),
    _cache()
{

}


pBuffer OperationCache::
        read( const Interval& I )
{
    static bool enable_cache = true;

    // cached samples doesn't count in samplesDesc if they are marked as invalid
    Intervals cached = _cache.samplesDesc() - _cache.invalid_samples();

    Interval ok = (cached & I).fetchFirstInterval();

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
                 b->getInterval().first,
                 b->getInterval().last);
    _cache.put(b);

    cached = _cache.samplesDesc() - _cache.invalid_samples();
    ok = cached & I;
    if (ok.first == I.first && ok.count())
    {
        return _cache.readFixedLength( ok );
    }

    missing = Intervals(I) - cached;
    _invalid_returns |= missing;
    return source()->readFixedLength( missing );
}


Intervals OperationCache::
        invalid_samples()
{
    Intervals c = _cache.invalid_samples();
    return c;
}


Intervals OperationCache::
        invalid_returns()
{
    return _invalid_returns;
}



} // namespace Signal
