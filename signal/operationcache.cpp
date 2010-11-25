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
    unsigned firstSample = I.first;

    Intervals cached = _cache.samplesDesc();
    cached -= _invalid_samples; // cached samples doesn't count if they are marked as invalid

    // read is only required to return firstSample, not the entire interval.
    // If the entire interval is needed for some other reason, cacheMiss can
    // be overloaded, such as in CwtFilter.
    Intervals need(firstSample, firstSample+1);
    need -= cached;

    // If we need something more, this is a cache miss
    return (bool)need;
}

pBuffer OperationCache::
        read( const Interval& I )
{
    if (!cacheMiss( I ))
    {
        // Don't need anything new, return cache
        pBuffer b = _cache.read( I );
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
