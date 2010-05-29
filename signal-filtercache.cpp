#include "signal-filtercache.h"
#include "signal-filteroperation.h"

namespace Signal {

FilterCache::
        FilterCache( pSource source )
:   Operation(source),
    _data()
{

}

pBuffer FilterCache::
        read( unsigned firstSample, unsigned numberOfSamples )
{
    // Check filters
    Operation* o = dynamic_cast<Operation*>(_source.get());

    BOOST_ASSERT(o);

    SamplesIntervalDescriptor cached = _data.samplesDesc();
    cached -= o->invalid_samples(); // cached samples doesn't count if they are marked as invalid

    SamplesIntervalDescriptor need(firstSample, numberOfSamples);
    need -= cached;

    if (need.isEmpty()) {
        // Don't need anything new, return cache
        return _data.read(firstSample, numberOfSamples);
    }

    pBuffer b = _source->read( firstSample, numberOfSamples );
    _data.put(b);
    return b;
}

} // namespace Signal
