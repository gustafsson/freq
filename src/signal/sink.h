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
class Sink: public Operation
{
public:
    Sink(): Operation(pOperation()) {}


    /// @overload Operation::read()
    virtual pBuffer read(const Interval& I) {
        BOOST_ASSERT(source());
        BOOST_ASSERT(I.count());

        pBuffer b = source()->read(I);

        // Check if read contains I.first
        BOOST_ASSERT(b->sample_offset <= I.first);
        BOOST_ASSERT(b->sample_offset + b->number_of_samples() > I.first);

        put(b);
        //_invalid_samples -= b->getInterval();
        return b;
    }


    /**
      A sink doesn't affect the buffer through 'read'.
      */
    virtual Signal::Intervals affected_samples() { return Signal::Intervals(); }


    /**
      If this Sink has recieved all expected_samples and is finished with its
      work, the caller may remove this Sink.
      */
    virtual bool deleteMe() { return !invalid_samples(); }
    virtual bool isUnderfed() { return false; }


    virtual void put(pBuffer) { throw std::logic_error(
            "Neither read nor put seems to have been overridden from Sink in " + vartype(*this) + "."); }

    /// @see invalid_samples()
    virtual void invalidate_samples(const Intervals& I) = 0;


    /**
      Example on how invalid_samples() is used:
      1. A brand new filter is inserted into the middle of the chain, it will
         affect some samples. To indicate that a change has been made but not
         yet processed invalid_samples is set to some equivalent non-empty
         values.
      2. ChainHead is notificed that something has changed and will query the
         chain for invalid samples. It will then issue invalidate_samples on
         all sinks connected to the ChainHead.
      3. Different sinks might react differently to invalidate_samples(...).
         The Heightmap::Collection for instance might only return a subset in
         invalid_samples() if those are the only samples that have been
         requested recently. Heightmap::Collection will return the other
         invalidated samples from invalid_samples() at a later occassion if
         they are requested by rendering.
         Signal::Playback will abort the playback and just stop, returning true
         from isFinished() and waiting to be deleted by the calling postsink
         and possibly recreated later.
    */
    virtual Intervals invalid_samples() = 0;


    static pBuffer put(Operation* receiver, pBuffer buffer) {
        pOperation s( new BufferSource(buffer));
        pOperation old = receiver->source();
        receiver->source(s);
        pBuffer r = receiver->read(buffer->getInterval());
        receiver->source(old);
        return r;
    }
};


} // namespace Signal

#endif // SIGNALSINK_H
