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
    Intervals cached = _cache.samplesDesc();
    cached -= _cache.invalid_samples();

    Interval ok = (Intervals(I) & cached).getInterval();

    if (ok.first == I.first && ok.valid() && enable_cache)
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

    Interval missing = I;
    if (ok.first != I.first && ok.valid())
        missing.last = ok.first;

    pBuffer b = readRaw( missing );
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
