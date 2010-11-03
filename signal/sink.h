#ifndef SIGNALSINK_H
#define SIGNALSINK_H

#include "signal/buffersource.h"
#include "signal/operation.h"
#include "signal/intervals.h"

namespace Signal {

/**
  Sink is just a convenience class, all sinks will be used through the
  Operation interface.

  An implementation may overload either read or put, whichever suits best.

  Any operation can be used as a sink in the sense that a sink is something
  that swallows data. The static method Sink::put can be used to send a buffer
  to an operation, the results may be discarded. This class hopefully makes it
  a little bit more clear by providing some convenient methods as examples.
  */
class Sink: public Operation
{
public:
    Sink(): Operation(pOperation()) {}


    /// @overload Operation::read()
    virtual pBuffer read(const Interval& I) {
        BOOST_ASSERT(source());
        pBuffer b = source()->read(I);
        put(b);
        _invalid_samples -= b->getInterval();
        return b;
    }


    /**
      In contrast to Operation::fetch_invalid_samples() this method doesn't
      clear _invalid_samples but just returns the result. A sink is different
      from an ordinary Operation in that it knows what data it wants, an
      Ordinary operation doesn't know what data it will be asked to process.

      @overload Operation::fetch_invalid_samples()
      */
    virtual Intervals fetch_invalid_samples() { return _invalid_samples; }


    /**
      A sink doesn't affect the buffer through 'read'.
      */
    virtual Signal::Intervals affected_samples() { return Signal::Intervals(); }


    /**
      If this Sink has recieved all expected_samples and is finished with its
      work, the caller may remove this Sink.
      */
    virtual bool isFinished() { return !fetch_invalid_samples(); }


    virtual void put(pBuffer) { throw std::logic_error(
            "Neither read nor put seems to have been overridden from Sink."); }

    /// @see fetch_invalid_samples()
    virtual void invalidate_samples(const Intervals& I) { _invalid_samples |= I; }


    static pBuffer put(Operation* receiver, pBuffer buffer) {
        pOperation s( new BufferSource(buffer));
        pOperation old = receiver->source();
        receiver->source(s);
        pBuffer r = receiver->read(buffer->getInterval());
        receiver->source(old);
        return r;
    }
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
