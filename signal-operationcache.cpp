#include "signal-operationcache.h"
#include "signal-filteroperation.h"

namespace Signal {

OperationCache::
        OperationCache( pSource source )
:   Operation(source)
{

}

pBuffer OperationCache::
        read( unsigned firstSample, unsigned numberOfSamples )
{
    // Check filters
    SamplesIntervalDescriptor cached = _data.samplesDesc();
    cached -= this->invalid_samples(); // cached samples doesn't count if they are marked as invalid

    SamplesIntervalDescriptor need(firstSample, firstSample+numberOfSamples);
    need -= cached;

    if (need.isEmpty()) {
        // Don't need anything new, return cache
        return _data.read( firstSample, numberOfSamples );
    }

    pBuffer b = readRaw( firstSample, numberOfSamples );
    _data.put(b);
    return b;
}

} // namespace Signal
