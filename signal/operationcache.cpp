#include "signal/operationcache.h"

static const bool D = false;

namespace Signal {

OperationCache::
        OperationCache( pOperation source )
:   Operation(source),
    _cache()
{

}


bool OperationCache::
        cacheMiss( const Interval& I )
{
    // cached samples doesn't count in samplesDesc if they are marked as invalid
    Intervals cached = _cache.samplesDesc();

    cached -= _cache.fetch_invalid_samples();

    // If we need something more, this is a cache miss
    return (bool)(Intervals(I) - cached);
}


pBuffer OperationCache::
        read( const Interval& I )
{
    static bool enable_cache = true;

    if (!cacheMiss( I ) && enable_cache)
    {
        // Don't need anything new, return cache
        pBuffer b = _cache.readFixedLength( I );
        if (D) TaskTimer("%s: cache [%u, %u] got [%u, %u]",
                     __FUNCTION__,
                     I.first,
                     I.last,
                     b->getInterval().first,
                     b->getInterval().last).suppressTiming();
        return b;
    }

    pBuffer b = readRaw( I );
    if (D) TaskTimer tt("%s: raw [%u, %u] got [%u, %u]",
                 __FUNCTION__,
                 I.first,
                 I.last,
                 b->getInterval().first,
                 b->getInterval().last);
    _cache.put(b);
    return b;
}

} // namespace Signal
