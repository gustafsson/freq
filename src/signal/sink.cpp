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
        SinkDesc(Signal::Operation::Ptr sink)
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


OperationDesc::Ptr SinkDesc::
        copy() const
{
    return OperationDesc::Ptr(new SinkDesc(sink_));
}


Operation::Ptr SinkDesc::
        createOperation(ComputingEngine* engine) const
{
    if (0 == engine)
        return sink_;

    return Operation::Ptr();
}


Operation::Ptr SinkDesc::
        sink() const
{
    return sink_;
}

}
