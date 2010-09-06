#ifndef SIGNALSINK_H
#define SIGNALSINK_H

#include "signal/buffersource.h"
#include "signal/operation.h"
#include "signal/intervals.h"

namespace Signal {

/**
  Any operation can be used as a sink in the sense that a sink is something
  that swallos data. This class hopefully makes it a little bit more clear
  by providing some convenient methods as examples.
  */
class Sink: public Operation
{
public:
    Sink():Operation(pSource()) {}

    virtual pBuffer read(const Interval& I) {
        BOOST_ASSERT(source());
        pBuffer b = source()->read(I);
        put(b);
        return b;
    }

    static void put(Operation* receiver, pBuffer buffer) {
        pSource s( new BufferSource(buffer));
        pSource old = receiver->source();
        receiver->source(s);
        receiver->read(buffer->getInterval());
        receiver->source(old);
    }

protected:
    virtual void put(pBuffer) { throw std::logic_error(
            "Neither read or put seems to have been overridden from Sink."); }
};

//{
//public:
//    virtual ~Sink() {}

//    /**
//      For some sinks it makes sense to reset, for some it doesn't.
//      */
//    virtual void reset() { _invalid_samples = Intervals(); }


//    /**
//      If this Sink has recieved all expected_samples and is finished with its
//      work, the caller may remove this Sink.
//      */
//    virtual bool isFinished() { return expected_samples().isEmpty(); }


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
