#include "signal/operation.h"

//#include <boost/serialization/export.hpp>
//BOOST_CLASS_EXPORT(Signal::Operation);

namespace Signal {

Operation::Operation(pOperation source )
:   _source( source ),
    _enabled( true ),
    _invalid_samples()
{
}


Intervals Operation::
        zeroed_samples()
{
    pOperation s = source();
    if (s)
        return s->zeroed_samples();
    return Intervals();
}


pBuffer Operation::
        read( const Interval& I )
{
    if (Intervals(I) - zeroed_samples())
        return source()->read( I );

    return zeros(I);
}


Operation* Operation::
        affecting_source( const Interval& I )
{
    if ((affected_samples() & I) || !source())
        return this;

    return source()->affecting_source(I);
}


// todo rename fetch_invalid_samples to read_invalid_samples
Intervals Operation::
        fetch_invalid_samples()
{
    Intervals r = _invalid_samples;

    Operation* o = source().get();
    if (0!=o)
    {
        r |= o->fetch_invalid_samples();
    }

    if (_invalid_samples)
        _invalid_samples = Intervals();

    return r;
}


Operation* Operation::
        root()
{
    if (source())
        return source()->root();

    return this;
}


} // namespace Signal
