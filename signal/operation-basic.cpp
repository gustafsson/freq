#include "signal/operation-basic.h"
#include <string.h>

namespace Signal {

    // OperationRemoveSection ///////////////////////////////////////////////////////////

OperationRemoveSection::
        OperationRemoveSection( pOperation source, IntervalType firstSample, IntervalType numberOfRemovedSamples )
:   Operation( source ),
    _firstSample( firstSample ),
    _numberOfRemovedSamples( numberOfRemovedSamples )
{}

pBuffer OperationRemoveSection::
        read( const Interval& I )
{
    IntervalType firstSample = I.first;
    IntervalType numberOfSamples = I.count();

    if (firstSample + numberOfSamples <= _firstSample )
    {
        return _source->read( I );
    }

    if (firstSample < _firstSample)
    {
        Interval I2(firstSample,_firstSample);

        return _source->read( I2 );
    }

    Interval I2(0,0);
    I2.first = firstSample + _numberOfRemovedSamples;
    I2.last = I2.first + numberOfSamples;

    pBuffer b = _source->read( I2 );
    b->sample_offset -= _numberOfRemovedSamples;
    return b;
}

IntervalType OperationRemoveSection::
        number_of_samples()
{
    IntervalType N = Operation::number_of_samples();
    if (N<_numberOfRemovedSamples)
        return 0;
    return N - _numberOfRemovedSamples;
}

    // OperationInsertSilence ///////////////////////////////////////////////////////////

OperationInsertSilence::
        OperationInsertSilence( pOperation source, IntervalType firstSample, IntervalType numberOfSilentSamples )
:   Operation( source ),
    _firstSample( firstSample ),
    _numberOfSilentSamples( numberOfSilentSamples )
{}


pBuffer OperationInsertSilence::
        read( const Interval& I )
{
    IntervalType firstSample = I.first;
    IntervalType numberOfSamples = I.count();

    if (firstSample + numberOfSamples <= _firstSample )
        return _source->read( I );

    if (firstSample < _firstSample)
        return _source->read( Interval(I.first, _firstSample - I.first) );

    if (firstSample >= _firstSample +  _numberOfSilentSamples) {
        pBuffer b = _source->read(
                Interval( I.first - _numberOfSilentSamples, I.first - _numberOfSilentSamples + numberOfSamples ));
        b->sample_offset += _numberOfSilentSamples;
        return b;
    }

    // Create silence
    IntervalType length = _numberOfSilentSamples - (firstSample - _firstSample);
    if ( length > numberOfSamples )
        length = numberOfSamples;

    return zeros(Signal::Interval(firstSample, firstSample+length));
}

IntervalType OperationInsertSilence::
        number_of_samples()
{
    return Operation::number_of_samples() + _numberOfSilentSamples;
}

// OperationSuperposition ///////////////////////////////////////////////////////////

OperationSuperposition::
        OperationSuperposition( pOperation source, pOperation source2 )
:   Operation( source ),
    _source2( source2 )
{
    if (_source->sample_rate() != _source2->sample_rate())
        throw std::invalid_argument("_source->sample_rate() != _source2->sample_rate()");
}

pBuffer OperationSuperposition::
        read( const Interval& I )
{
    pBuffer a = _source->read( I );
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
