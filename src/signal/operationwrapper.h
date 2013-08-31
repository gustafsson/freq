#ifndef SIGNAL_OPERATIONWRAPPER_H
#define SIGNAL_OPERATIONWRAPPER_H

#include "operation.h"

namespace Signal {


/**
 * @brief The OperationWrapper class should wrap all calls to another Operation.
 *
 * It should be a transparent no-op operation if the other operation is null.
 */
class OperationWrapper: public Operation {
public:
    OperationWrapper(Operation::Ptr wrap);

    void setWrappedOperation(Operation::Ptr wrap);

    virtual Signal::pBuffer process(Signal::pBuffer b);

private:
    Operation::Ptr wrap_;
};


/**
 * @brief The OperationDescWrapper class should behave as another OperationDesc.
 *
 * It should ensure that when the wrapped operation is changed all instanciated
 * operations must be recreated.
 *
 * It should behave as a transparent operation if no operation is wrapped.
 */
class OperationDescWrapper: public OperationDesc {
public:
    OperationDescWrapper(OperationDesc::Ptr wrap=OperationDesc::Ptr());

    void setWrappedOperationDesc(OperationDesc::Ptr wrap);
    OperationDesc::Ptr getWrappedOperationDesc();

    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    virtual Signal::Interval requiredInterval( const Signal::Intervals& I, Signal::Interval* expectedOutput ) const;
    virtual Interval affectedInterval( const Interval& I ) const;
    virtual Intervals affectedInterval( const Intervals& I ) const;
    virtual OperationDesc::Ptr copy() const;
    virtual Operation::Ptr createOperation(ComputingEngine* engine) const;
    virtual Extent extent() const;
    virtual Operation::Ptr recreateOperation(Operation::Ptr hint, ComputingEngine* engine) const;
    virtual QString toString() const;
    virtual int getNumberOfSources() const;
    virtual bool operator==(const OperationDesc& d) const;

protected:
    virtual OperationWrapper* createOperationWrapper(ComputingEngine* engine, Operation::Ptr wrapped) const;

private:
    OperationDesc::Ptr wrap_;
    typedef std::map<ComputingEngine*, Operation::WeakPtr> OperationMap;
    boost::shared_ptr<OperationMap> map_;

public:
    static void test();
};

} // namespace Signal

#endif // SIGNAL_OPERATIONWRAPPER_H
