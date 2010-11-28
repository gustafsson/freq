#ifndef SIGNALOPERATION_H
#define SIGNALOPERATION_H

#include "signal/source.h"
#include "signal/intervals.h"
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/shared_ptr.hpp>

// For debug info while serializing
#include <demangle.h>

namespace Signal {

typedef boost::shared_ptr<class Operation> pOperation;

/**
A Signal::Operation is a Signal::Source which reads data from another 
Signal::Source and performs some operation on that data before returning it to
the caller.
 */
class Operation: public SourceBase
{
public:
    /**
      This constructor by itself creates a dummy Operation that redirects 'read'
      to its _source.
      */
    Operation( pOperation source );


    /**
      Overloaded from Source. The default implementation of read is to read
      from source().

      Note that read doesn't have to be called. See affected_samples().
      */
    virtual pBuffer read( const Interval& I );
    virtual float sample_rate() { return source()->sample_rate(); }  /// @see read(const Interval&)
    virtual long unsigned number_of_samples() { return source()->number_of_samples(); } /// @see read(const Interval&)


    /**
      What makes an operation is that it processes a signal that actually comes
      from somewhere else. This 'somewhere else' is defined by another
      Operation. In the end there is some kind of operation that doesn't rely
      on some source but provides all data by itself.
      */
    virtual pOperation source() const { return _source; }
    virtual void source(pOperation v) { _source=v; } /// @see source()


    /**
      'affected_samples' describes where it is possible that
        'source()->readFixedLength( I ) != readFixedLength( I )'
      '!affected_samples' describes where it is guaranteed that
        'source()->readFixedLength( I ) == readFixedLength( I )'

      A filter is allowed to be passive in some parts and nonpassive in others.
      'affected_samples' describes where. 'affected_samples' is allowed to
      change over time as well, but to make sure that changed intervals are
      requested for _invalid_samples must be changed accordingly.
      _invalid_samples is then copied and cleared by 'fetch_invalid_samples'.

      As default all samples are possibly affected by an Operation.
      */
    virtual Signal::Intervals affected_samples() { return Signal::Intervals::Intervals_ALL; }


    /**
      These samples are definitely set to 0 by the filter. As default returns
      source()->zeroed_samples if source() is not null, or no samples if
      source() is null.

      @remarks zeroed_samples is _assumed_ (but never checked) to be a subset
      of Signal::Operation::affected_samples().
      */
    virtual Signal::Intervals zeroed_samples();


    /**
      Since many sources might leave many samples unaffected it is unnecessary
      to process through them. Just skip to the next source that will actually
      do something. Then read from that source with readFixedLength(). Do not
      read with read() as it might return some other interval that clash with
      affected_samples() even if the requested interval doesn't clash.

      Some operations require that they are processed even though they won't
      affect their samples. These are operations that processes the data
      further somewhere else. Such operations can either overload
      affecting_source and return themselves as 'return this', or if the
      information on what Interval that is processed is enough they may
      overload affecting_source, use the Interval and still do
      'return source()->affecting_source()'.

      Note that affecting_source may not even be called if a simple chain of
      read() is used instead.

      affecting_source will return source()->affecting_source() if enabled()
      is false.
      */
    virtual Operation* affecting_source( const Interval& I );


    /**
      Fetches and clears invalid samples recursively.
      Returns _invalid_samples merged with source()->invalid_samples().

      Example on how invalid_samples() is used:
      1. A brand new filter is inserted into the middle of the chain, it will
         affect some samples. To indicate that a change has been made but not
         yet processed invalid_samples is set to some equivalent non-empty
         values.
      2. Worker is notificed that something has changed and will query the
         chain for invalid samples. It will then issue invalidate_samples on
         all sinks connected to the worker.
      3. Different sinks might react differently to invalidate_samples(...).
         The Heightmap::Collection for instance might only return a subset in
         invalid_samples() if those are the only samples that have been
         requested recently. Heightmap::Collection will return the other
         invalidated samples from invalid_samples() at a later occassion if
         they are requested by rendering.
         Signal::Playback will abort the playback and just stop, returning true
         from isFinished() and waiting to be deleted by the calling postsink
         and possibly recreated later.
      4. fetch_invalid_samples is not const with respect to Operation, because
         fetch_invalid_samples clears _invalid_samples after each call.
    */
    virtual Intervals fetch_invalid_samples();


    /**
      An operation can be disabled. If it is not enabled any call to read must
      return source()->read();
      */
    virtual bool enabled() { return _enabled; }
    virtual void enabled(bool value) { _enabled = value; }


    Operation* root();

protected:
    pOperation _source; /// @see Operation::source()
    bool _enabled; /// @see Operation::enabled()

    /**
      _invalid_samples describes which samples that should be re-read off from
      Operation. It is initialized to an empty interval and can be
      used by an implementaion to say that the previous results are out of
      date.

      @see fetch_invalid_samples()
      */
    Intervals _invalid_samples;

private:
    Operation() {} // used by serialization
    friend class boost::serialization::access;
    template<class archive>
    void serialize(archive& ar, const unsigned int /*version*/)
    {
        if (_source)
            TaskTimer("*source is: %s", vartype(*_source).c_str()).suppressTiming();

        ar & BOOST_SERIALIZATION_NVP(_source);
    }
};


/**
  FinalSource overloads Operation and disables support for Operation::source().
  It is in functionality almost equivalent to SourceBase but can be shipped
  around as pOperation.
  */
class FinalSource: public Operation
{
public:
    FinalSource() : Operation(pOperation()) {}

    virtual pBuffer read( const Interval& I ) = 0;
    virtual float sample_rate() = 0;
    virtual long unsigned number_of_samples() = 0;

private:
    virtual pOperation source() const { return pOperation(); }
    virtual void source(pOperation)   { throw std::logic_error("Invalid call"); }
};

} // namespace Signal

#endif // SIGNALOPERATION_H
