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

    // TODO overload these as well, Intervals need to be translated
    // virtual Intervals affected_samples() { return Intervals(); }
    // virtual Intervals invalid_samples();
    // virtual void invalidate_samples(const Intervals& I) { _invalid_samples |= I; }
private:
    IntervalType _firstSample, _numberOfRemovedSamples;
};

class OperationInsertSilence: public Operation
{
public:
    OperationInsertSilence( pOperation source, IntervalType firstSample, IntervalType numberOfSilentSamples );

    virtual pBuffer read( const Interval& I );
    virtual IntervalType number_of_samples();

    // TODO overload these as well, Intervals need to be translated
    // virtual Intervals affected_samples() { return Intervals(); }
    // virtual Intervals invalid_samples();
    // virtual void invalidate_samples(const Intervals& I) { _invalid_samples |= I; }
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
