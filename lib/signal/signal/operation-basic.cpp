#include "signal/operation-basic.h"
#include <string.h>

namespace Signal {

    // OperationSetSilent  /////////////////////////////////////////////////////////////////

OperationSetSilent::Operation::
        Operation(const Interval &section)
    :
      section_(section)
{}


pBuffer OperationSetSilent::Operation::
        process (pBuffer b)
{
    Signal::Interval i = section_ & b->getInterval ();

    if (i) {
        Buffer zero(i, b->sample_rate(), b->number_of_channels ());
        *b |= zero;
    }

    return b;
}


OperationSetSilent::
        OperationSetSilent( const Signal::Interval& section )
    :
      section_(section)
{
}


Interval OperationSetSilent::
        requiredInterval( const Interval& I, Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Interval OperationSetSilent::
        affectedInterval( const Interval& I ) const
{
    return I;
}


OperationDesc::ptr OperationSetSilent::
        copy() const
{
    return OperationDesc::ptr(new OperationSetSilent(section_));
}


Signal::Operation::ptr OperationSetSilent::
        createOperation(ComputingEngine*) const
{
    return Signal::Operation::ptr(new OperationSetSilent::Operation(section_));
}


QString OperationSetSilent::
        toString() const
{
    std::stringstream ss;
    ss << "Clear section " << section_;
    return QString::fromStdString (ss.str());
}

} // namespace Signal
