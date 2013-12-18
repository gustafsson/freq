#include "signal/operation.h"

#include "demangle.h"

#include <boost/foreach.hpp>


//#define TIME_OPERATION
#define TIME_OPERATION if(0)

//#define TIME_OPERATION_LINE(x) TIME(x)
#define TIME_OPERATION_LINE(x) x


namespace Signal {


void Operation::
        test(Ptr o, OperationDesc* desc)
{
    Interval I(40,70), expectedOutput;
    Interval ri = desc->requiredInterval (I, &expectedOutput);
    EXCEPTION_ASSERT( (Interval(I.first, I.first+1) & ri).count() > 0 );
    pBuffer d(new Buffer(ri, 40, 7));
    pBuffer p = write1(o)->process (d);
    EXCEPTION_ASSERT_EQUALS( p->getInterval (), expectedOutput );
}


Intervals OperationDesc::
        affectedInterval( const Intervals& I ) const
{
    Intervals A;
    BOOST_FOREACH(const Interval& i, I) {
            A |= affectedInterval(i);
    }

    return A;
}


OperationDesc::Extent OperationDesc::
        extent() const
{
    return Extent();
}


Operation::Ptr OperationDesc::
        recreateOperation(Operation::Ptr /*hint*/, ComputingEngine* engine) const
{
    return createOperation(engine);
}


QString OperationDesc::
        toString() const
{
    return vartype(*this).c_str();
}


int OperationDesc::
        getNumberOfSources() const
{
    return 1;
}


void OperationDesc::
        setInvalidator(Signal::Processing::IInvalidator::Ptr invalidator)
{
    invalidator_ = invalidator;
}


bool OperationDesc::
        operator==(const OperationDesc& d) const
{
    &d?void():void(); // suppress unused argument warning
    return typeid(*this) == typeid(d);
}


std::ostream& operator << (std::ostream& os, const OperationDesc& d) {
    return os << d.toString().toStdString ();
}


void OperationDesc::
        deprecateCache(Signal::Intervals what) const volatile
{
    Signal::Processing::IInvalidator::Ptr invalidator = ReadPtr(this)->invalidator_;

    if (invalidator)
        read1(invalidator)->deprecateCache(what);
}


DeprecatedOperation::DeprecatedOperation(pOperation s )
{
    source( s );
}


DeprecatedOperation::
        ~DeprecatedOperation()
{
    source( pOperation() );
}


DeprecatedOperation::
        DeprecatedOperation( const DeprecatedOperation& o )
            :
            SourceBase( o )
{
    *this = o;
}


DeprecatedOperation& DeprecatedOperation::
        operator=(const DeprecatedOperation& o )
{
    DeprecatedOperation::source( o.DeprecatedOperation::source() );
    return *this;
}


void DeprecatedOperation::
        source(pOperation v)
{
    if (_source)
        _source->_outputs.erase( this );

    _source=v;

    if (_source)
        _source->_outputs.insert( this );
}


Intervals DeprecatedOperation::
        affected_samples()
{
    return Intervals::Intervals_ALL;
}


Intervals DeprecatedOperation::
        zeroed_samples_recursive()
{
    Intervals I = zeroed_samples();
    if (_source)
        I |= translate_interval( _source->zeroed_samples_recursive() );
    return I;
}


Intervals DeprecatedOperation::
        zeroed_samples()
{
    return Intervals(number_of_samples(), Interval::IntervalType_MAX);
}


std::string DeprecatedOperation::
        name()
{
    return vartype(*this);
}


pBuffer DeprecatedOperation::
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


IntervalType DeprecatedOperation::
        number_of_samples()
{
    return _source ? _source->number_of_samples() : 0;
}


float DeprecatedOperation::
        length()
{
    float L = SourceBase::length();
    float D = 0.f; // _source ? _source->length() - _source->SourceBase::length() : 0;
    return std::max(0.f, L + D);
}


DeprecatedOperation* DeprecatedOperation::
        affecting_source( const Interval& I )
{
    if ((affected_samples() & I) || !_source)
        return this;

    return _source->affecting_source( I );
}


void DeprecatedOperation::
        invalidate_samples(const Intervals& I)
{
    if (!I)
        return;

    BOOST_FOREACH( DeprecatedOperation* p, _outputs )
    {
        EXCEPTION_ASSERT( 0 != p );
        p->invalidate_samples( p->translate_interval( I ));
    }
}


DeprecatedOperation* DeprecatedOperation::
        root()
{
    if (_source)
        return _source->root();

    return this;
}


bool DeprecatedOperation::
        hasSource(DeprecatedOperation*s)
{
    if (this == s)
        return true;
    if (_source)
        return _source->hasSource( s );
    return false;
}


pOperation DeprecatedOperation::
        findParentOfSource(pOperation start, pOperation source)
{
    if (start->source() == source)
        return start;
    if (start->source())
        return findParentOfSource(start->source(), source);

    return pOperation();
}


Intervals DeprecatedOperation::
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


std::string DeprecatedOperation::
        toString()
{
    std::string s = toStringSkipSource();

    if (_source)
        s += "\n" + _source->toString();

    return s;
}


std::string DeprecatedOperation::
        toStringSkipSource()
{
    std::string s = name();

    std::string n = DeprecatedOperation::name();
    if (s != n)
    {
        s += " (";
        s += n;
        s += ")";
    }

    return s;
}


std::string DeprecatedOperation::
        parentsToString()
{
    std::string s = name();

    std::string n = DeprecatedOperation::name();
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
    BOOST_FOREACH( DeprecatedOperation* p, _outputs )
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
