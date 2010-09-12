#ifndef SIGNALSINK_H
#define SIGNALSINK_H

#include "signal/buffersource.h"
#include "signal/operation.h"
#include "signal/intervals.h"

namespace Signal {

/**
  Any operation can be used as a sink in the sense that a sink is something
  that swallows data. This class hopefully makes it a little bit more clear
  by providing some convenient methods as examples.
  */
class Sink: public Operation
{
public:
    Sink(): Operation(pOperation()) {}

    /**
      If this Sink has recieved all expected_samples and is finished with its
      work, the caller may remove this Sink.
      */
    virtual bool isFinished() { return invalid_samples().isEmpty(); }

    virtual pBuffer read(const Interval& I) {
        BOOST_ASSERT(source());
        pBuffer b = source()->read(I);
        put(b);
        _invalid_samples |= b->getInterval();
        return b;
    }

    static void put(Operation* receiver, pBuffer buffer) {
        pOperation s( new BufferSource(buffer));
        pOperation old = receiver->source();
        receiver->source(s);
        receiver->read(buffer->getInterval());
        receiver->source(old);
    }

    virtual void put(pBuffer) { throw std::logic_error(
            "Neither read nor put seems to have been overridden from Sink."); }
protected:
};

//{
//public:
//    virtual ~Sink() {}

//    /**
//      For some sinks it makes sense to reset, for some it doesn't.
//      */
//    virtual void reset() { _invalid_samples = Intervals(); }


//    /**
//      If a Sink should do something special when it has received all Buffers,
//      do it in onFinished(). onFinished() may be invoked more than once.
//      */
//    virtual void onFinished() {}


//    // TODO virtual bool isUnderfed()


//    /**
//      By telling the sink through 'add_expected_samples' how much data the sink
//      can expect to recieve it is possible for the sink to perform some
//      optimizations (such as buffering input before starting playing a sound).
//      */
//    virtual Intervals expected_samples() { return _invalid_samples; }
//    virtual void add_expected_samples( const Intervals& s ) { _invalid_samples |= s; }
//};

} // namespace Signal

#endif // SIGNALSINK_H
