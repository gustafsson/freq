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
    Signal::Interval i = section_ & b->getInterval ();

    if (i) {
        Buffer zero(i, b->sample_rate(), b->number_of_channels ());
        *b |= zero;
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


QString OperationSetSilent::
        toString() const
{
    std::stringstream ss;
    ss << "Clear section " << section_;
    return QString::fromStdString (ss.str());
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


} // namespace Signal
