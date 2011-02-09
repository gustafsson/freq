#include "signal/operation.h"

#include <demangle.h>

namespace Signal {

Operation::Operation(pOperation source )
:   _source( source ),
    _enabled( true ),
    _invalid_samples()
{
}


Signal::Intervals Operation::
        affected_samples()
{
    IntervalType it = number_of_samples();
    return (_enabled && it)?Intervals(0, it ):Intervals();
}


Intervals Operation::
        zeroed_samples_recursive()
{
    Intervals I = zeroed_samples();
    if (_source)
        I |= translate_interval( _source->zeroed_samples_recursive() );
    return I;
}


Intervals Operation::
        zeroed_samples()
{
    return Intervals();
}


pBuffer Operation::
        read( const Interval& I )
{
    if (Intervals(I) - zeroed_samples())
        return _source->read( I );

    return zeros(I);
}


Operation* Operation::
        affecting_source( const Interval& I )
{
    if ((affected_samples() & I) || !_source)
        return this;

    return _source->affecting_source( I );
}


// todo rename fetch_invalid_samples to read_invalid_samples
Intervals Operation::
        fetch_invalid_samples()
{
//    TaskInfo tt("%s::fetch_invalid_samples, _invalid_samples=%s",
//                vartype(*this).c_str(), _invalid_samples.toString().c_str());
    Intervals r = _invalid_samples;

    Operation* o = _source.get();
    if (0!=o)
    {
        r |= translate_interval(o->fetch_invalid_samples());
    }

    if (_invalid_samples)
        _invalid_samples = Intervals();

    return r;
}


Operation* Operation::
        root()
{
    if (_source)
        return _source->root();

    return this;
}


std::string Operation::
        toString()
{
    std::string s = vartype(*this);
    if (_source)
        s += "\n" + _source->toString();
    return s;
}


Signal::Intervals Operation::
        affected_samples_until(pOperation stop)
{
    Signal::Intervals I = affected_samples();
    if (this!=stop.get() && _source)
        I |= translate_interval( _source->affected_samples_until( stop ) );
    return I;
}


} // namespace Signal
