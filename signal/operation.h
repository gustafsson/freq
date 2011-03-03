#ifndef SIGNALOPERATION_H
#define SIGNALOPERATION_H

//signal
#include "source.h"
#include "intervals.h"

// boost
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/shared_ptr.hpp>

// gpumisc
// For debug info while serializing
#include <demangle.h>

#include <set>

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
    ~Operation();

    virtual std::string name();

    /**
      Overloaded from Source. The default implementation of read is to read
      from _source. Or return zeros if _source is null.

      Note that read doesn't have to be called. See affected_samples().
      */
    virtual pBuffer read( const Interval& I );

    /**
      sample_rate is invalid to call if _source is null.

      @see read(const Interval&)
      */
    virtual float sample_rate() { return _source->sample_rate(); }
    virtual IntervalType number_of_samples(); /// @see read(const Interval&)

    virtual unsigned num_channels() { return _source->num_channels(); }
    virtual void set_channel(unsigned c) { if(_source) _source->set_channel(c); }
    virtual unsigned get_channel() { return _source->get_channel(); }


    /**
      What makes an operation is that it processes a signal that actually comes
      from somewhere else. This 'somewhere else' is defined by another
      Operation. In the end there is a special kind of operation (FinalSource)
      that doesn't rely on its source() but provides all data by itself.
      */
    virtual pOperation source() const { return _source; }
    virtual void source(pOperation v); /// @see source(), outputs()


    /**
      outputs() points to all Operations that has this as their source.

      May be empty if no Operation currently has this as source.

      The parent-source relation is setup and maintained by source(pOperation);

      outputs() are used by Operation::invalidate_samples.
      */
    virtual std::set<Operation*> outputs() { return _outputs; }

    /**
      'affected_samples' describes where it is possible that
        'source()->readFixedLength( I ) != readFixedLength( I )'
      '~affected_samples' describes where it is guaranteed that
        'source()->readFixedLength( I ) == readFixedLength( I )'

      A filter is allowed to be passive in some parts and nonpassive in others.
      'affected_samples' describes where. 'affected_samples' is allowed to
      change over time as well.

      As default all samples are possibly affected by an Operation.
      */
    virtual Signal::Intervals affected_samples();


    /**
      Merges zeroed_samples with _source->zeroed_samples() if _source is not
      null.
      */
    virtual Signal::Intervals zeroed_samples_recursive();


    /**
      These samples are definitely set to 0 by this filter. As default returns
      no samples.

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

//      affecting_source will return source()->affecting_source() if enabled()
//      is false since affected_samples() is empty if enabled() is false.

      This also skips wrapper containers that doesn't do anything themselves.

      Returns 'this' if this Operation does something.
      */
    virtual Operation* affecting_source( const Interval& I );


    /**
      @see OperationSubOperations
      */
    virtual Signal::Intervals affected_samples_until(pOperation stop);


    /**
      An implementation of Operation needs to overload this if the samples are
      moved in some way.

      Example: OperationRemoveSection removes some samples from the signal.
      Let's say the section to remove is [10,20) then we have:
        'translate_interval([0,10))' -> '[0,10)'
        'translate_interval([10,20))' -> '[)'
        'translate_interval([20,30))' -> '[10,20)'
        'translate_interval([0,30))' -> '[0,20)'

      The default implementation returns the same interval.

      translate_interval is used by Operation::zeroed_samples_recursive and
      Operation::affected_samples_until and Operation::invalidate_samples.

      @see OperationRemoveSection, OperationInsertSilence, zeroed_samples
      */
    virtual Signal::Intervals translate_interval(Signal::Intervals I) { return I; }


    /**
      invalidate_samples propagates information that something has changed to
      all outputs. In the end some outputs should be sinks and register that
      the data they've got so far is no longer valid.
      */
    virtual void invalidate_samples(const Intervals& I);


    /**
      An operation can be disabled. If it is not enabled any call to read must
      return source()->read();
      */
//    virtual bool enabled() { return _enabled; }
//    virtual void enabled(bool value) { _enabled = value; }


    Operation* root();
    virtual bool hasSource(Operation*s);

    /**
      If 's' has multiple outputs, find the output that leads to 'this' and
      return the parent of 's' from that trace.
      */
    static pOperation findParentOfSource(pOperation start, pOperation source);
    static Signal::Intervals affectedDiff(pOperation source1, pOperation source2);

    virtual std::string toString();
    virtual std::string parentsToString();

private:
    std::set<Operation*> _outputs; /// @see Operation::parent()
    pOperation _source; /// @see Operation::source()
    //bool _enabled; /// @see Operation::enabled()

    friend class boost::serialization::access;
    Operation() /// only used by deserialization, call Operation(pOperation) instead
    {}
    template<class archive>
    void serialize(archive& ar, const unsigned int /*version*/)
    {
        TaskInfo ti("Serializing %s, source: %s",
                    name().c_str(), _source.get()?_source->name().c_str():0);
        ti.tt().partlyDone();

        ar & BOOST_SERIALIZATION_NVP(_source);

        Operation::source(_source);
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
    virtual IntervalType number_of_samples() = 0;

    virtual unsigned num_channels() = 0;
	virtual void set_channel(unsigned c) = 0;
    virtual unsigned get_channel() = 0;

    virtual Signal::Intervals zeroed_samples();

private:
    virtual pOperation source() const { return pOperation(); }
    virtual void source(pOperation)   { throw std::logic_error("Invalid call"); }
};

} // namespace Signal

#endif // SIGNALOPERATION_H
