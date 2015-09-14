#include "signal/operation.h"

#include "demangle.h"

#include <boost/foreach.hpp>


//#define TIME_OPERATION
#define TIME_OPERATION if(0)

//#define TIME_OPERATION_LINE(x) TIME(x)
#define TIME_OPERATION_LINE(x) x


namespace Signal {


void Operation::
        test(ptr o, OperationDesc* desc)
{
    Interval I(40,70), expectedOutput;
    Interval ri = desc->requiredInterval (I, &expectedOutput);
    EXCEPTION_ASSERT( (Interval(I.first, I.first+1) & ri).count() > 0 );
    pBuffer d(new Buffer(ri, 40, 7));
    pBuffer p = o->process (d);
    EXCEPTION_ASSERT_EQUALS( p->getInterval (), expectedOutput );
}


OperationDesc::Extent OperationDesc::
        extent() const
{
    return Extent();
}


QString OperationDesc::
        toString() const
{
    return vartype(*this).c_str();
}


void OperationDesc::
        setInvalidator(Signal::Processing::IInvalidator::ptr invalidator)
{
    invalidator_ = invalidator;
}


Signal::Processing::IInvalidator::ptr OperationDesc::
        getInvalidator() const
{
    return invalidator_;
}


bool OperationDesc::
        operator==(const OperationDesc& d) const
{
    //&d?void():void(); // suppress unused argument warning
    return typeid(*this) == typeid(d);
}


std::ostream& operator << (std::ostream& os, const OperationDesc& d) {
    return os << d.toString().toStdString ();
}

} // namespace Signal
