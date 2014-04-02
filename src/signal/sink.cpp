#include "sink.h"

namespace Signal {


Sink::
    ~Sink()
{

}


bool Sink::
        deleteMe()
{
    return false;
}


bool Sink::
        isUnderfed()
{
    return false;
}


SinkDesc::
        SinkDesc(Signal::Operation::ptr sink)
    :
      sink_(sink)
{

}


Interval SinkDesc::
        requiredInterval( const Interval& I, Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Interval SinkDesc::
        affectedInterval( const Interval& I ) const
{
    return I;
}


OperationDesc::ptr SinkDesc::
        copy() const
{
    return OperationDesc::ptr(new SinkDesc(sink_));
}


Operation::ptr SinkDesc::
        createOperation(ComputingEngine* engine) const
{
    if (0 == engine)
        return sink_;

    return Operation::ptr();
}


Operation::ptr SinkDesc::
        sink() const
{
    return sink_;
}

}
