#include "signal/operation.h"

#include <demangle.h>

#include <boost/foreach.hpp>


//#define TIME_OPERATION
#define TIME_OPERATION if(0)

//#define TIME_OPERATION_LINE(x) TIME(x)
#define TIME_OPERATION_LINE(x) x


namespace Signal {

Operation::Operation(pOperation s )
{
    source( s );
}


Operation::
        ~Operation()
{
    source( pOperation() );
}


Operation::
        Operation( const Operation& o )
            :
            SourceBase( o )
{
    *this = o;
}


Operation& Operation::
        operator=(const Operation& o )
{
    Operation::source( o.Operation::source() );
    return *this;
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


Intervals Operation::
        affected_samples()
{
    return Intervals::Intervals_ALL;
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
    {
        TIME_OPERATION TaskTimer tt("%s.%s(%s) from %s",
                      vartype(*this).c_str(), __FUNCTION__ ,
                      I.toString().c_str(),
                      vartype(*_source).c_str());

        return _source->read( I );
    }

    TIME_OPERATION TaskTimer tt("%s.%s(%s) zeros",
                  vartype(*this).c_str(), __FUNCTION__ ,
                  I.toString().c_str());
    return zeros(I);
}


IntervalType Operation::
        number_of_samples()
{
    return _source ? _source->number_of_samples() : 0;
}


float Operation::
        length()
{
    float L = SourceBase::length();
    float D = 0.f; // _source ? _source->length() - _source->SourceBase::length() : 0;
    return std::max(0.f, L + D);
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
    if (!I)
        return;

    BOOST_FOREACH( Operation* p, _outputs )
    {
        EXCEPTION_ASSERT( 0 != p );
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
    if (start->source() == source)
        return start;
    if (start->source())
        return findParentOfSource(start->source(), source);

    return pOperation();
}


Intervals Operation::
        affectedDiff(pOperation source1, pOperation source2)
{
    Intervals new_data( 0, source1->number_of_samples() );
    Intervals old_data( 0, source2->number_of_samples() );
    Intervals invalid = new_data | old_data;

    Intervals was_zeros = source1->zeroed_samples_recursive();
    Intervals new_zeros = source2->zeroed_samples_recursive();
    Intervals still_zeros = was_zeros & new_zeros;
    invalid -= still_zeros;

    Intervals affected_samples_until_source2;
    for (pOperation o = source1; o && o!=source2; o = o->source())
        affected_samples_until_source2 |= o->affected_samples();

    Intervals affected_samples_until_source1;
    for (pOperation o = source2; o && o!=source1; o = o->source())
        affected_samples_until_source1 |= o->affected_samples();

    invalid &= affected_samples_until_source2;
    invalid &= affected_samples_until_source1;

    invalid |= source1->affected_samples();
    invalid |= source2->affected_samples();

    return invalid;
}


std::string Operation::
        toString()
{
    std::string s = toStringSkipSource();

    if (_source)
        s += "\n" + _source->toString();

    return s;
}


std::string Operation::
        toStringSkipSource()
{
    std::string s = name();

    std::string n = Operation::name();
    if (s != n)
    {
        s += " (";
        s += n;
        s += ")";
    }

    return s;
}


std::string Operation::
        parentsToString()
{
    std::string s = name();

    std::string n = Operation::name();
    if (s != n)
    {
        s += " (";
        s += n;
        s += ")";
    }

    std::stringstream ss;
    ss << s;
    if (1 < _outputs.size())
        ss << " (" << _outputs.size() << " parents)";

    unsigned i = 1;
    BOOST_FOREACH( Operation* p, _outputs )
    {
        ss << std::endl;
        if (_outputs.size() > 1)
        {
            if (i>1)
                ss << std::endl;
            ss << i << " in " << name() << ": ";
        }
        i++;
        ss << p->parentsToString();
    }
    return ss.str();
}


Intervals FinalSource::
        zeroed_samples()
{
    IntervalType N = number_of_samples();
    Intervals r = Intervals::Intervals_ALL;
    if (N)
        r -= Interval(0, N);
    return r;
}

} // namespace Signal
