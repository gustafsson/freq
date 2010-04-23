#include "signal-operation.h"

namespace Signal {

Operation::
Operation(pSource source )
:   _source( source )
{
}

unsigned Operation::
sample_rate()
{
    return _source->sample_rate();
}

unsigned Operation::
number_of_samples()
{
    return _source->number_of_samples();
}

SamplesIntervalDescriptor Operation::
updateInvalidSamples()
{
    Operation* o = dynamic_cast<Operation*>(_source.get());

    if (0!=o)
        _invalid_samples |= o->updateInvalidSamples();

    return _invalid_samples;
}

pSource Operation::first_source(pSource start)
{
    Operation* o = dynamic_cast<Operation*>(start.get());
    if (o)
        return first_source(o->source());

    return start;
}

} // namespace Signal
