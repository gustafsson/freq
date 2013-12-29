#include "signal/operation-basic.h"
#include <string.h>

namespace Signal {

    // OperationSetSilent  /////////////////////////////////////////////////////////////////

OperationSetSilent::Operation::
        Operation(const Interval &section)
    :
      section_(section)
{}


pBuffer OperationSetSilent::Operation::
        process (pBuffer b)
{
    Signal::Intervals I = section_;
    I &= b->getInterval ();

    foreach (Signal::Interval i, I) {
        pBuffer zero( new Buffer(i, b->sample_rate(), b->number_of_channels ()) );
        *b |= *zero;
    }

    return b;
}


OperationSetSilent::
        OperationSetSilent( const Signal::Interval& section )
    :
      section_(section)
{
}


Interval OperationSetSilent::
        requiredInterval( const Interval& I, Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Interval OperationSetSilent::
        affectedInterval( const Interval& I ) const
{
    return I;
}


OperationDesc::Ptr OperationSetSilent::
        copy() const
{
    return OperationDesc::Ptr(new OperationSetSilent(section_));
}


Signal::Operation::Ptr OperationSetSilent::
        createOperation(ComputingEngine*) const
{
    return Signal::Operation::Ptr(new OperationSetSilent::Operation(section_));
}

    // OperationSetSilent  /////////////////////////////////////////////////////////////////

DeprecatedOperationSetSilent::
        DeprecatedOperationSetSilent( pOperation source, const Signal::Interval& section )
:   DeprecatedOperation( source ),
    section_( section )
{
}

std::string DeprecatedOperationSetSilent::
        name()
{
    float fs = source()?sample_rate():0.f;
    std::stringstream ss;
    if (fs > 0)
        ss << "Clear section [" << section_.first/fs << ", " << section_.last/fs << ") s";
    else
        ss << "Clear section " << section_;
    return ss.str();
}

pBuffer DeprecatedOperationSetSilent::
        read( const Interval& I )
{
    Interval t = I & section_;
    if ( t.first == I.first && t.count() )
        return zeros( t );
    return source()->readFixedLength( (I - section_).fetchFirstInterval() );
}


    // OperationRemoveSection ///////////////////////////////////////////////////////////

OperationRemoveSection::
        OperationRemoveSection( pOperation source, Interval section )
:   DeprecatedOperation( source ),
    section_(section)
{
    EXCEPTION_ASSERT(section_.count());
}

pBuffer OperationRemoveSection::
        read( const Interval& I )
{
    const Interval neg(Interval::IntervalType_MIN, 0);
    Interval r = translate_interval_inverse(I).fetchFirstInterval ();

    if ((I & neg) && !(r & neg))
        return zeros(I & neg);

    pBuffer b = source()->readFixedLength(r);
    b->set_sample_offset ( I.first );

    return b;
}

IntervalType OperationRemoveSection::
        number_of_samples()
{
    Interval pos(0, Interval::IntervalType_MAX);
    Interval sectionp = pos & section_;
    IntervalType N = DeprecatedOperation::number_of_samples();
    if (!sectionp.count ())
        return N;

    if (N<=sectionp.last)
        return std::min(N, sectionp.first);

    return N - sectionp.count();
}

Intervals OperationRemoveSection::
        affected_samples()
{
    const Interval pos(0, Interval::IntervalType_MAX);
    const Interval neg(Interval::IntervalType_MIN, 0);

    Intervals I;

    if (section_ & pos)
        I |= pos & Interval(section_.first, pos.last);
    if (section_ & neg)
        I |= neg & Interval(pos.first, section_.last);

    return I;
}

Signal::Intervals OperationRemoveSection::
        translate_interval(Signal::Intervals I)
{
    Interval pos(0, Interval::IntervalType_MAX);
    Interval neg(Interval::IntervalType_MIN, 0);
    Interval rightkeep(section_.last, Interval::IntervalType_MAX);
    Interval leftkeep(Interval::IntervalType_MIN, section_.first);

    Intervals left = (I & leftkeep) <<= (section_ & neg).count ();
    Intervals right = (I & rightkeep) >>= (section_ & pos).count ();

    left |= I & rightkeep;
    right |= I & leftkeep;
    left &= neg;
    right &= pos;

    return left | right;
}

Signal::Intervals OperationRemoveSection::
        translate_interval_inverse(Signal::Intervals I)
{
    Interval pos(0, Interval::IntervalType_MAX);
    Interval neg(Interval::IntervalType_MIN, 0);
    Interval rightmove(section_.first, Interval::IntervalType_MAX);
    Interval leftmove(Interval::IntervalType_MIN, section_.last);

    Intervals left = (leftmove & neg & I) >>= (section_ & neg).count();
    Intervals right = (rightmove & pos & I) <<= (section_ & pos).count();

    Interval rightkeep(section_.last, Interval::IntervalType_MAX);
    Interval leftkeep(Interval::IntervalType_MIN, section_.first);

    left |= rightkeep & neg & I;
    right |= leftkeep & pos & I;

    Intervals r = left | right;
    return r;
}


    // OperationInsertSilence ///////////////////////////////////////////////////////////

OperationInsertSilence::
        OperationInsertSilence( pOperation source, Interval section )
:   DeprecatedOperation( source ),
    section_( section )
{
    EXCEPTION_ASSERT( section.first >= 0 );
}


pBuffer OperationInsertSilence::
        read( const Interval& I )
{
    if (I.last <= section_.first )
        return source()->readFixedLength( I );

    if (I.first < section_.first)
        return source()->readFixedLength( Interval(I.first, section_.first) );

    if (I.first >= section_.last) {
        pBuffer b = source()->readFixedLength(
                (Intervals( I ) >>= section_.count()).spannedInterval());
        b->set_sample_offset ( b->sample_offset () + section_.count() );
        return b;
    }

    // Create silence
    Interval silence = I;
    if ( silence.last > section_.last)
        silence.last = section_.last;

    return zeros(silence);
}

IntervalType OperationInsertSilence::
        number_of_samples()
{
    UnsignedIntervalType N = DeprecatedOperation::number_of_samples();
    if (0 <= section_.first && N <= (UnsignedIntervalType)section_.first)
        return section_.last;
    if (N + section_.count() < N)
        return Interval::IntervalType_MAX;
    return N + section_.count();
}

Intervals OperationInsertSilence::
        affected_samples()
{
    return Signal::Interval(section_.first, Signal::Interval::IntervalType_MAX);
}

Signal::Intervals OperationInsertSilence::
        translate_interval(Signal::Intervals I)
{
    Signal::Intervals beginning, ending;

    if (section_.first)
        beginning = Signal::Interval( 0, section_.first );

    ending = Signal::Interval( section_.first, Signal::Interval::IntervalType_MAX );

    return (I&beginning) | ((I&ending) <<= section_.count());
}

Signal::Intervals OperationInsertSilence::
        translate_interval_inverse(Signal::Intervals I)
{
    Signal::Intervals beginning, ending;

    if (section_.first)
        beginning = Signal::Interval( 0, section_.first );

    ending = Signal::Interval(section_.last, Signal::Interval::IntervalType_MAX);

    return (I&beginning) | ((I&ending) >>= section_.count());
}


// OperationSuperposition ///////////////////////////////////////////////////////////

OperationSuperposition::
        OperationSuperposition( pOperation source, pOperation source2 )
:   DeprecatedOperation( source ),
    _source2( source2 )
{
//    if (Operation::source()->sample_rate() != _source2->sample_rate())
//        throw std::invalid_argument("source->sample_rate() != source2->sample_rate()");
}


std::string OperationSuperposition::
        name()
{
    return _name.empty() ? DeprecatedOperation::name() : _name;
}


void OperationSuperposition::
        name(std::string n)
{
    _name = n;
}


pBuffer OperationSuperposition::
        read( const Interval& I )
{   
    pBuffer a = source()->read( I );
    pBuffer b = _source2->read( I );

    return superPosition(a, b, false);
}


pBuffer OperationSuperposition::
        superPosition( pBuffer a, pBuffer b, bool inclusive )
{
    EXCEPTION_ASSERT( a->sample_rate () == b->sample_rate () );
    EXCEPTION_ASSERT( a->number_of_channels () == b->number_of_channels () );
    Interval I;
    if (inclusive)
        I = a->getInterval ().spanned ( b->getInterval () );
    else
        I = a->getInterval () & b->getInterval ();

    pBuffer r(new Buffer( I.first, I.count (), a->sample_rate (), a->number_of_channels ()));
    *r += *a;
    *r += *b;
    return r;
}


IntervalType OperationSuperposition::
        number_of_samples()
{
    return std::max( DeprecatedOperation::number_of_samples(), _source2->number_of_samples() );
}


unsigned OperationSuperposition::
        num_channels()
{
    return std::max( DeprecatedOperation::num_channels(), _source2->num_channels() );
}


Intervals OperationSuperposition::
        zeroed_samples()
{
    return source()->zeroed_samples() & _source2->zeroed_samples();
}


Intervals OperationSuperposition::
        affected_samples()
{
    return _source2->affected_samples();
}


Signal::Intervals OperationSuperposition::
        zeroed_samples_recursive()
{
    return DeprecatedOperation::zeroed_samples_recursive() & _source2->zeroed_samples();
}


// OperationAddChannels ///////////////////////////////////////////////////////////


OperationAddChannels::
        OperationAddChannels( pOperation source, pOperation source2 )
    :
    DeprecatedOperation(source),
    source2_(source2)
{
}


pBuffer OperationAddChannels::
        read( const Interval& I )
{
    pBuffer b1 = source()->read( I );
    pBuffer b2 = source2_->readFixedLength( b1->getInterval() );
    int c1 = b1->number_of_channels();
    int c2 = b2->number_of_channels();

    pBuffer r(new Buffer(b1->getInterval(), b1->sample_rate(), c1+c2));

    for (int c=0; c<c1; ++c)
        *r->getChannel(c) |= *b1->getChannel(c);
    for (int c=c1; c<c1+c2; ++c)
        *r->getChannel(c) |= *b2->getChannel(c - c1);

    return r;
}


IntervalType OperationAddChannels::
        number_of_samples()
{
    return std::max(source()->number_of_samples(), source2_->number_of_samples());
}


unsigned OperationAddChannels::
        num_channels()
{
    return source()->num_channels() + source2_->num_channels();
}



// OperationSuperPositionChannels ///////////////////////////////////////////////////////////


OperationSuperpositionChannels::
        OperationSuperpositionChannels( pOperation source )
    :
    DeprecatedOperation(source)
{
}


pBuffer OperationSuperpositionChannels::
        read( const Interval& I )
{
    pBuffer b = source()->read( I );
    pBuffer r( new Buffer(b->sample_offset (), b->number_of_samples (), b->sample_rate (), 1));
    pMonoBuffer sum = r->getChannel (0);
    for (unsigned c = 0; c < b->number_of_channels (); c++)
        *sum |= *b->getChannel (c);
    return r;
}


Signal::Intervals OperationSuperpositionChannels::
        affected_samples()
{
    return getInterval();
}

} // namespace Signal
