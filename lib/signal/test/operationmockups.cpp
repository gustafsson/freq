#include "operationmockups.h"

namespace Test {


Signal::pBuffer TransparentOperation::
        process(Signal::pBuffer b)
{
    return b;
}


Signal::Interval TransparentOperationDesc::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;

    return I;
}


Signal::Interval TransparentOperationDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    return I;
}


Signal::Operation::ptr TransparentOperationDesc::
        createOperation(Signal::ComputingEngine*) const
{
    return Signal::Operation::ptr( new TransparentOperation );
}


Signal::OperationDesc::ptr TransparentOperationDesc::
        copy() const
{
    return OperationDesc::ptr( new TransparentOperationDesc );
}


EmptySource::
        EmptySource(float sample_rate, int number_of_channels)
    :
      BufferSource(Signal::pBuffer(new Signal::Buffer(Signal::Interval(), sample_rate, number_of_channels)))
{
}


} // namespace Test
