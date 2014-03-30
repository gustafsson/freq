#ifndef SIGNAL_OPERATIONWRAPPER_H
#define SIGNAL_OPERATIONWRAPPER_H

#include "operation.h"
#include "signal/processing/iinvalidator.h"

namespace Signal {

/**
 * @brief The OperationDescWrapper class should behave as another OperationDesc.
 *
 * It should ensure that instantiated operations are recreated when the wrapped
 * operation is changed.
 *
 * It should behave as a transparent operation if no operation is wrapped.
 *
 * It should allow new opertions without blocking while processing existing operations.
 */
class OperationDescWrapper: public OperationDesc {
public:
    OperationDescWrapper(OperationDesc::Ptr wrap=OperationDesc::Ptr());

    /**
     * @brief setWrappedOperationDesc makes this OperationDesc behave as a new
     * operation.
     */
    void setWrappedOperationDesc2(OperationDesc::Ptr wrap);
    OperationDesc::Ptr getWrappedOperationDesc() const;

    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    virtual Interval affectedInterval( const Interval& I ) const;
    virtual OperationDesc::Ptr copy() const;
    virtual Operation::Ptr createOperation(ComputingEngine* engine) const;
    virtual Extent extent() const;
    virtual QString toString() const;
    virtual bool operator==(const OperationDesc& d) const;

private:
    OperationDesc::Ptr wrap_;

public:
    static void test();
};

} // namespace Signal

#endif // SIGNAL_OPERATIONWRAPPER_H
