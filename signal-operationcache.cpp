#include "signal-operationcache.h"
#include "signal-filteroperation.h"

static const bool D = false;

namespace Signal {

OperationCache::
        OperationCache( pSource source )
:   Operation(source),
    _cache( SinkSource::AcceptStrategy_ACCEPT_ALL)
{

}

bool OperationCache::
        cacheMiss(unsigned firstSample, unsigned /*numberOfSamples*/)
{
    SamplesIntervalDescriptor cached = _cache.samplesDesc();
    cached -= this->invalid_samples(); // cached samples doesn't count if they are marked as invalid

    // read is only required to return firstSample, not the entire interval.
    // If the entire interval is needed for some other reason, cacheMiss can
    // be overloaded, such as in FilterOperation.
    SamplesIntervalDescriptor need(firstSample, firstSample+1);
    need -= cached;

    // If we need something more, this is a cache miss
    return !need.isEmpty();
}

pBuffer OperationCache::
        read( unsigned firstSample, unsigned numberOfSamples )
{
    if (!cacheMiss(firstSample, numberOfSamples ))
    {
        // Don't need anything new, return cache
        pBuffer b = _cache.read( firstSample, numberOfSamples );
        if (D) TaskTimer("%s: cache [%u, %u] got [%u, %u]",
                     __FUNCTION__,
                     firstSample,
                     firstSample+numberOfSamples,
                     b->getInterval().first,
                     b->getInterval().last).suppressTiming();
        return b;
    }

    pBuffer b = readRaw( firstSample, numberOfSamples );
    if (D) TaskTimer tt("%s: raw [%u, %u] got [%u, %u]",
                 __FUNCTION__,
                 firstSample,
                 firstSample+numberOfSamples,
                 b->getInterval().first,
                 b->getInterval().last);
    _cache.put(b);
    return b;
}

} // namespace Signal
