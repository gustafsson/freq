#include "signal/operation.h"

#include <demangle.h>

#include <boost/foreach.hpp>

namespace Signal {

Operation::Operation(pOperation s )
//:   _enabled( true ) // TODO remove _enabled
{
    source( s );
}


Operation::
        ~Operation()
{
    source( pOperation() );
}


void Operation::
        source(pOperation v)
{
    if (_source)
        _source->_outputs.erase( this );

    _source=v;

    if (_source)
        _source->_outputs.insert( this );
}


Signal::Intervals Operation::
        affected_samples()
{
    return getInterval();
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
    return Intervals(number_of_samples(), Interval::IntervalType_MAX);
}


std::string Operation::
        name()
{
    return vartype(*this);
}


pBuffer Operation::
        read( const Interval& I )
{
    if (_source && Intervals(I) - zeroed_samples())
        return _source->read( I );

    return zeros(I);
}


IntervalType Operation::
        number_of_samples()
{
    return _source ? _source->number_of_samples() : 0;
}


Operation* Operation::
        affecting_source( const Interval& I )
{
    if ((affected_samples() & I) || !_source)
        return this;

    return _source->affecting_source( I );
}


void Operation::
        invalidate_samples(const Intervals& I)
{
    BOOST_FOREACH( Operation* p, _outputs )
    {
        p->invalidate_samples( p->translate_interval( I ));
    }
}


Operation* Operation::
        root()
{
    if (_source)
        return _source->root();

    return this;
}


bool Operation::
        hasSource(Operation*s)
{
    if (this == s)
        return true;
    if (_source)
        return _source->hasSource( s );
    return false;
}


pOperation Operation::
        findParentOfSource(pOperation start, pOperation source)
{
    if (start->_source == source)
        return start;
    if (start->_source)
        return findParentOfSource(start->_source, source);

    return pOperation();
}


Signal::Intervals Operation::
        affectedDiff(pOperation source1, pOperation source2)
{
    Signal::Intervals new_data( 0, source1->number_of_samples() );
    Signal::Intervals old_data( 0, source2->number_of_samples() );
    Signal::Intervals invalid = new_data | old_data;

    Signal::Intervals was_zeros = source1->zeroed_samples_recursive();
    Signal::Intervals new_zeros = source2->zeroed_samples_recursive();
    Signal::Intervals still_zeros = was_zeros & new_zeros;
    invalid -= still_zeros;

    invalid &= source1->affected_samples_until( source2 );
    invalid &= source2->affected_samples_until( source1 );

    invalid |= source1->affected_samples();
    invalid |= source2->affected_samples();

    return invalid;
}


std::string Operation::
        toString()
{
    std::string s = name();

    if (_source)
        s += "\n" + _source->toString();

    return s;
}


std::string Operation::
        parentsToString()
{
    std::stringstream ss;
    ss << name() << " (" << _outputs.size() << " parent" << (_outputs.size()==1?"":"s") << ")";
    unsigned i = 0;
    BOOST_FOREACH( Operation* p, _outputs )
    {
        ss << std::endl;
        if (_outputs.size())
            ss << i++ << ": ";
        ss << p->parentsToString();
    }
    return ss.str();
}


Signal::Intervals Operation::
        affected_samples_until(pOperation stop)
{
    Signal::Intervals I;
    if (this!=stop.get())
    {
        I = affected_samples();
        if (_source)
            I |= translate_interval( _source->affected_samples_until( stop ) );
    }
    return I;
}


Signal::Intervals FinalSource::
        zeroed_samples()
{
    IntervalType N = number_of_samples();
    Signal::Intervals r = Signal::Intervals::Intervals_ALL;
    if (N)
        r -= Signal::Interval(0, N);
    return r;
}

} // namespace Signal
