#ifndef SIGNALSINK_H
#define SIGNALSINK_H

#include "buffersource.h"
#include "operation.h"
#include "intervals.h"

namespace Signal {

/**
  A sink is different from an ordinary Operation in that it knows what data it
  wants, an Ordinary operation doesn't know what data it will be asked to
  process.

  An implementation may overload either read or put, whichever suits best.

  Any operation can be used as a sink in the sense that a sink is something
  that swallows data. The static method Sink::put can be used to send a buffer
  to an operation, the results may be discarded. This class hopefully makes it
  a little bit more clear by providing some convenient methods as examples.
 */
class Sink: public Signal::Operation
{
public:
    virtual ~Sink();

    // Overloaded from Signal::Operation
    Signal::pBuffer process(Signal::pBuffer b) { put(b); return b; }

    /**
      If this Sink has recieved all expected_samples and is finished with its
      work, the caller may remove this Sink.
      */
    virtual bool deleteMe();
    virtual bool isUnderfed();
    virtual void put(pBuffer) = 0;
};


class SinkDesc: public Signal::OperationDesc
{
public:
    SinkDesc(Signal::Operation::ptr sink);

    Interval requiredInterval( const Interval& I, Interval* expectedOutput ) const;
    Interval affectedInterval( const Interval& I ) const;
    OperationDesc::ptr copy() const;
    Operation::ptr createOperation(ComputingEngine* engine) const;
    Operation::ptr sink() const;

private:
    Signal::Operation::ptr sink_;
};

} // namespace Signal

#endif // SIGNALSINK_H
