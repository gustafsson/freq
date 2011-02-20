#include "signal/operation-basic.h"
#include <string.h>

namespace Signal {

    // OperationRemoveSection ///////////////////////////////////////////////////////////

OperationRemoveSection::
        OperationRemoveSection( pOperation source, Interval section )
:   Operation( source ),
    section_(section)
{
    BOOST_ASSERT(section_.count());
}

pBuffer OperationRemoveSection::
        read( const Interval& I )
{
    if (I.last <= section_.first )
    {
        return source()->readFixedLength( I );
    }

    if (I.first < section_.first)
    {
        return source()->readFixedLength( Interval(I.first, section_.first) );
    }

    if (I.first + section_.count() + 1 < I.first)
    {
        return zeros(I);
    }

    pBuffer b = source()->readFixedLength( Intervals(I) << section_.count() );
    b->sample_offset -= section_.count();

    return b;
}

IntervalType OperationRemoveSection::
        number_of_samples()
{
    IntervalType N = Operation::number_of_samples();
    if (N<section_.last)
        return std::min(N, section_.first);
    return N - section_.count();
}

Intervals OperationRemoveSection::
        affected_samples()
{
    return Signal::Interval(section_.first, Signal::Interval::IntervalType_MAX);
}

Signal::Intervals OperationRemoveSection::
        translate_interval(Signal::Intervals I)
{
    Signal::Intervals beginning, ending;

    if (section_.first)
        beginning = Signal::Interval( 0, section_.first );

    if (section_.last < Signal::Interval::IntervalType_MAX)
        ending = Signal::Interval( section_.last, Signal::Interval::IntervalType_MAX );

    return (I&beginning) | ((I&ending) >> section_.count());
}

Signal::Intervals OperationRemoveSection::
        translate_interval_inverse(Signal::Intervals I)
{
    Signal::Intervals beginning, ending;

    if (section_.first)
        beginning = Signal::Interval( 0, section_.first );

    ending = Signal::Interval( section_.first, Signal::Interval::IntervalType_MAX );

    return (I&beginning) | ((I&ending) << section_.count());
}

    // OperationInsertSilence ///////////////////////////////////////////////////////////

OperationInsertSilence::
        OperationInsertSilence( pOperation source, Interval section )
:   Operation( source ),
    section_( section )
{
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
                Intervals( I ) >> section_.count());
        b->sample_offset += section_.count();
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
    IntervalType N = Operation::number_of_samples();
    if (N <= section_.first)
        return N;
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

    return (I&beginning) | ((I&ending) << section_.count());
}

Signal::Intervals OperationInsertSilence::
        translate_interval_inverse(Signal::Intervals I)
{
    Signal::Intervals beginning, ending;

    if (section_.first)
        beginning = Signal::Interval( 0, section_.first );

    ending = Signal::Interval(section_.last, Signal::Interval::IntervalType_MAX);

    return (I&beginning) | ((I&ending) >> section_.count());
}


// OperationSuperposition ///////////////////////////////////////////////////////////

OperationSuperposition::
        OperationSuperposition( pOperation source, pOperation source2 )
:   Operation( source ),
    _source2( source2 )
{
    if (Operation::source()->sample_rate() != _source2->sample_rate())
        throw std::invalid_argument("source->sample_rate() != source2->sample_rate()");
}

pBuffer OperationSuperposition::
        read( const Interval& I )
{
    pBuffer a = source()->read( I );
    pBuffer b = _source2->read( I );

    IntervalType offset = std::max( (IntervalType)a->sample_offset, (IntervalType)b->sample_offset );
    IntervalType length = std::min(
            (IntervalType)a->sample_offset + a->number_of_samples(),
            (IntervalType)b->sample_offset + b->number_of_samples() );
    length -= offset;

    pBuffer r(new Buffer( offset, length, sample_rate() ));

    float *pa = a->waveform_data()->getCpuMemory();
    float *pb = b->waveform_data()->getCpuMemory();
    float *pr = r->waveform_data()->getCpuMemory();

    pa += (IntervalType)(r->sample_offset - a->sample_offset);
    pb += (IntervalType)(r->sample_offset - b->sample_offset);

    for (unsigned i=0; i<r->number_of_samples(); i++)
        pr[i] = pa[i] + pb[i];

    return r;
}

} // namespace Signal
