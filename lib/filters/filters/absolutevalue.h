#ifndef FILTERS_ABSOLUTEVALUE_H
#define FILTERS_ABSOLUTEVALUE_H

#include "signal/operation.h"

namespace Filters {

class AbsoluteValue : public Signal::Operation
{
public:
    Signal::pBuffer process(Signal::pBuffer b);
};


/**
 * @brief The AbsoluteValueDesc class should compute the absolute value of a signal.
 */
class AbsoluteValueDesc : public Signal::OperationDesc
{
public:
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    Signal::OperationDesc::ptr copy() const;
    Signal::Operation::ptr createOperation(Signal::ComputingEngine* engine) const;

public:
    static void test();
};

} // namespace Filters

#endif // FILTERS_ABSOLUTEVALUE_H
