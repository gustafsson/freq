#ifndef SIGNAL_OLDOPERATIONWRAPPER_H
#define SIGNAL_OLDOPERATIONWRAPPER_H

#include "operation.h"
#include "volatileptr.h"

namespace Signal {

/**
 * @brief The OldOperationIntervals class should guess the DeprecatedOperation
 * equivalent of requiredInterval and
 */
class OldOperationIntervals: public VolatilePtr<OldOperationIntervals> {
public:

    Interval guessAffectedInterval(Interval input);
    Interval guessRequiredSamples(Interval output);

    void teach(Interval process_output, Interval requested_input);
    void requestInfo(Interval request);

private:
    IntervalType enlarge_;
    Interval last_request_;
};


/**
 * @brief The OldOperationWrapper class should use a DeprectatedOperation to
 * compute the result of processing a step.
 */
class OldOperationWrapper: public Operation {
public:
    OldOperationWrapper(pOperation old_operation);

private:
    pBuffer process(pBuffer b);

    pOperation old_operation_;

public:
    static void test();
};


/**
 * @brief The OldOperationDescWrapper class should represent an instance of
 * DeprecatedOperation in Processing::Chain.
 */
class OldOperationDescWrapper: public OperationDesc
{
public:
    OldOperationDescWrapper(pOperation old_operation);

    pOperation old_operation() { return old_operation_; }

private:
    Interval requiredInterval( const Interval& I, Interval* expectedOutput ) const;
    Interval affectedInterval( const Interval& I ) const;
    OperationDesc::Ptr copy() const;
    Operation::Ptr createOperation(ComputingEngine* engine) const;
    Extent extent() const;
    QString toString() const;

    pOperation old_operation_;

public:
    static void test();
};

} // namespace Signal

#endif // SIGNAL_OLDOPERATIONWRAPPER_H
