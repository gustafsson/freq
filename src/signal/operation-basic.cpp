#include "signal/operation-basic.h"
#include <string.h>

namespace Signal {

    // OperationSetSilent  /////////////////////////////////////////////////////////////////
OperationSetSilent::
        OperationSetSilent( pOperation source, const Signal::Interval& section )
:   Operation( source ),
    section_( section )
{
}

std::string OperationSetSilent::
        name()
{
    float fs = source()?sample_rate():1;
    std::stringstream ss;
    ss << "Clear section [" << section_.first/fs << ", " << section_.last/fs << ") s";
    return ss.str();
}

pBuffer OperationSetSilent::
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
:   Operation( source ),
    section_(section)
{
    BOOST_ASSERT(section_.count());
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
    IntervalType N = Operation::number_of_samples();
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

    Intervals left = (I & leftkeep) << (section_ & neg).count ();
    Intervals right = (I & rightkeep) >> (section_ & pos).count ();

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

    Intervals left = (leftmove & neg & I) >> (section_ & neg).count();
    Intervals right = (rightmove & pos & I) << (section_ & pos).count();

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
:   Operation( source ),
    section_( section )
{
    BOOST_ASSERT( section.first >= 0 );
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
                (Intervals( I ) >> section_.count()).spannedInterval());
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
//    if (Operation::source()->sample_rate() != _source2->sample_rate())
//        throw std::invalid_argument("source->sample_rate() != source2->sample_rate()");
}


std::string OperationSuperposition::
        name()
{
    return _name.empty() ? Operation::name() : _name;
}


void OperationSuperposition::
        name(std::string n)
{
    _name = n;
}


pBuffer OperationSuperposition::
        read( const Interval& I )
{   
    pBuffer a, b;

    if ( Operation::get_channel() == get_channel() )
        a = source()->read( I );
    if ( _source2->get_channel() == get_channel() )
        b = _source2->read( I );

    if ( a && b )
        return superPosition(a, b);
    else if (a)
        return a;
    else if (b)
        return b;
    else
        return Operation::zeros( I );
}


pBuffer OperationSuperposition::
        superPosition( pBuffer a, pBuffer b )
{
    BOOST_ASSERT( a->sample_rate == b->sample_rate );
    IntervalType offset = std::max(a->sample_offset.asInteger(), b->sample_offset.asInteger());
    IntervalType length = std::min(
            a->sample_offset.asInteger() + a->number_of_samples(),
            b->sample_offset.asInteger() + b->number_of_samples() );
    length -= offset;

    pBuffer r(new Buffer( offset, length, a->sample_rate ));

    float *pa = a->waveform_data()->getCpuMemory();
    float *pb = b->waveform_data()->getCpuMemory();
    float *pr = r->waveform_data()->getCpuMemory();

    pa += (r->sample_offset - a->sample_offset).asInteger();
    pb += (r->sample_offset - b->sample_offset).asInteger();

    for (unsigned i=0; i<r->number_of_samples(); i++)
        pr[i] = pa[i] + pb[i];

    return r;
}


IntervalType OperationSuperposition::
        number_of_samples()
{
    return std::max( Operation::number_of_samples(), _source2->number_of_samples() );
}


unsigned OperationSuperposition::
        num_channels()
{
    return std::max( Operation::num_channels(), _source2->num_channels() );
}


void OperationSuperposition::
        set_channel(unsigned c)
{
    if (c < _source2->num_channels())
        _source2->set_channel(c);

    if (c < Operation::num_channels())
        Operation::set_channel(c);
}


unsigned OperationSuperposition::
        get_channel()
{
    return std::max( Operation::get_channel(), _source2->get_channel() );
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
    return Operation::zeroed_samples_recursive() & _source2->zeroed_samples();
}


// OperationAddChannels ///////////////////////////////////////////////////////////


OperationAddChannels::
        OperationAddChannels( pOperation source, pOperation source2 )
    :
    Operation(source),
    source2_(source2),
    current_channel_(0)
{
}


pBuffer OperationAddChannels::
        read( const Interval& I )
{
    if (current_channel_< source()->num_channels())
        return source()->read( I );
    else
        return source2_->read( I );
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


void OperationAddChannels::
        set_channel(unsigned c)
{
    BOOST_ASSERT( c < num_channels() );
    if (c < source()->num_channels())
        source()->set_channel(c);
    else
        source2_->set_channel(c - source()->num_channels());
    current_channel_ = c;
}



// OperationSuperPositionChannels ///////////////////////////////////////////////////////////


OperationSuperpositionChannels::
        OperationSuperpositionChannels( pOperation source )
    :
    Operation(source)
{
}


pBuffer OperationSuperpositionChannels::
        read( const Interval& I )
{
    pBuffer sum;
    for (unsigned c = 0; c < source()->num_channels(); c++)
    {
        source()->set_channel( c );
        pBuffer b = source()->read( I );
        if (sum)
            sum = OperationSuperposition::superPosition(sum, b);
        else
            sum = b;
    }
    return sum;
}


void OperationSuperpositionChannels::
        set_channel(unsigned c)
{
    BOOST_ASSERT( c == 0 );
}


Signal::Intervals OperationSuperpositionChannels::
        affected_samples()
{
    return getInterval();
}

} // namespace Signal
