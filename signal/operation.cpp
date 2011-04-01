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


Operation::
        Operation( const Operation& o )
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


pBuffer Operation::
        readFixedLengthAllChannels( const Interval& I )
{
    if (1 >= num_channels())
        return readFixedLength( I );

    unsigned current_channel = this->get_channel();
    pBuffer b( new Signal::Buffer(I.first, I.count(), sample_rate(), this->num_channels() ));

    float* dst = b->waveform_data()->getCpuMemory();
    for (unsigned i=0; i<num_channels(); ++i)
    {
        this->set_channel( i );
        Signal::pBuffer r = readFixedLength( I );
        float* src = r->waveform_data()->getCpuMemory();
        memcpy( dst + i*I.count(), src, I.count()*sizeof(float));
    }
    this->set_channel( current_channel );

    return b;
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
    Intervals J = I & getInterval();

    if (!J)
        return;

    BOOST_FOREACH( Operation* p, _outputs )
    {
        p->invalidate_samples( p->translate_interval( J ));
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
    std::string s = name();

    std::string n = Operation::name();
    if (s != n)
    {
        s += " (";
        s += n;
        s += ")";
    }

    if (_source)
        s += "\n" + _source->toString();

    return s;
}


std::string Operation::
        parentsToString()
{
    std::stringstream ss;
    ss << name();
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
