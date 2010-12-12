#ifndef SIGNALOPERATIONBASIC_H
#define SIGNALOPERATIONBASIC_H

#include "signal/operation.h"

namespace Signal {

class OperationRemoveSection: public Operation
{
public:
    OperationRemoveSection( pOperation source, IntervalType firstSample, IntervalType numberOfRemovedSamples );

    virtual pBuffer read( const Interval& I );
    virtual IntervalType number_of_samples();

    virtual Intervals affected_samples() { return Signal::Interval::Interval_ALL; }
    virtual Intervals zeroed_samples() { return Operation::zeroed_samples() >> _firstSample; }
    virtual Intervals fetch_invalid_samples() { return Operation::fetch_invalid_samples( ) >> _firstSample; }
private:

    IntervalType _firstSample, _numberOfRemovedSamples;
};

class OperationInsertSilence: public Operation
{
public:
    OperationInsertSilence( pOperation source, IntervalType firstSample, IntervalType numberOfSilentSamples );

    virtual pBuffer read( const Interval& I );
    virtual IntervalType number_of_samples();

    virtual Intervals affected_samples() { return Signal::Interval(_firstSample, Signal::Interval::IntervalType_MAX); }
    virtual Intervals zeroed_samples() { return Signal::Interval(_firstSample, _firstSample+_numberOfSilentSamples ); }
    virtual Intervals fetch_invalid_samples() { return Operation::fetch_invalid_samples( ) << _firstSample; }
private:
    IntervalType _firstSample, _numberOfSilentSamples;
};

class OperationSuperposition: public Operation
{
public:
    OperationSuperposition( pOperation source, pOperation source2 );

    virtual pBuffer read( const Interval& I );

    virtual pOperation source2() const { return _source2; }

private:
    pOperation _source2;
};

} // namespace Signal

#endif // SIGNALOPERATIONBASIC_H
