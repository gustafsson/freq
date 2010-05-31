#include "signal-operation.h"
#include "signal-filteroperation.h"

namespace Signal {

Operation::Operation(pSource source )
:   _source( source ),
    _invalid_samples()
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
        invalid_samples()
{
    Operation* o = dynamic_cast<Operation*>(_source.get());

    SamplesIntervalDescriptor r = _invalid_samples;

    if (0!=o)
        r |= o->invalid_samples();

    return r;
}

pSource Operation::
        first_source(pSource start)
{
    Operation* o = dynamic_cast<Operation*>(start.get());
    if (o)
        return first_source(o->source());

    return start;
}

pSource Operation::
        fast_source(pSource start)
{
    pSource r = start;
    pSource itr = start;

    while(true)
    {
        Operation* o = dynamic_cast<Operation*>(itr.get());
        if (!o)
            break;

        FilterOperation* f = dynamic_cast<FilterOperation*>(itr.get());
        if (f)
            r = f->source();

        itr = o->source();
    }

    return r;
}

} // namespace Signal
