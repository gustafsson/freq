#include "signal/operation.h"
#include "tfr/cwtfilter.h"

namespace Signal {

Operation::Operation(pOperation source )
:   _source( source ),
    _invalid_samples()
{
}

#include <stdio.h> // todo remove
Intervals Operation::
        invalid_samples()
{
    Operation* o = dynamic_cast<Operation*>(source().get());

    Intervals r = _invalid_samples;

    if (0!=o)
    {
        r |= o->invalid_samples();
    }

    return r;
}

pOperation Operation::
        first_source(pOperation start)
{
    Operation* o = dynamic_cast<Operation*>(start.get());
    if (o)
        return first_source(o->source());

    return start;
}

pOperation Operation::
        fast_source(pOperation start)
{
    pOperation r = start;
    pOperation itr = start;

    while(true)
    {
        Operation* o = dynamic_cast<Operation*>(itr.get());
        if (!o)
            break;

        Tfr::CwtFilter* f = dynamic_cast<Tfr::CwtFilter*>(itr.get());
        if (f)
            r = f->source();

        itr = o->source();
    }

    return r;
}

} // namespace Signal
