#ifndef TEST_OPERATIONMOCKUPS_H
#define TEST_OPERATIONMOCKUPS_H

#include "signal/operation.h"

namespace Test {


class TransparentOperation: public Signal::Operation {
public:
    virtual Signal::pBuffer process(Signal::pBuffer b);
    virtual Signal::Interval requiredInterval( const Signal::Interval& I );
};

class TransparentOperationDesc: public Signal::OperationDesc {
public:
    virtual Signal::Operation::Ptr createOperation(Signal::ComputingEngine*) const;
    virtual OperationDesc::Ptr copy() const;
    virtual QString toString() const;
    virtual bool operator==(const OperationDesc& d) const;
};

} // namespace Test

#endif // TEST_OPERATIONMOCKUPS_H
