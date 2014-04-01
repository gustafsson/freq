#ifndef TEST_OPERATIONMOCKUPS_H
#define TEST_OPERATIONMOCKUPS_H

#include "signal/operation.h"
#include "signal/buffersource.h"

namespace Test {


class TransparentOperation: public Signal::Operation {
public:
    virtual Signal::pBuffer process(Signal::pBuffer b);
};

class TransparentOperationDesc: public Signal::OperationDesc {
public:
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    virtual Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    virtual Signal::Operation::ptr createOperation(Signal::ComputingEngine*) const;
    virtual OperationDesc::ptr copy() const;
};


class EmptySource: public Signal::BufferSource {
public:
    EmptySource(float sample_rate=1.f, int number_of_channels=1);
};

} // namespace Test

#endif // TEST_OPERATIONMOCKUPS_H
