#ifndef SIGNALOPERATION_H
#define SIGNALOPERATION_H

//signal
#include "source.h"
#include "intervals.h"
#include "poperation.h"

// boost
#include <boost/serialization/nvp.hpp>

// gpumisc
// For debug info while serializing
#include <demangle.h>
#include <TaskTimer.h>

// std
#include <set>

// QString
#include <QString>

namespace Signal {

class ComputingEngine;

class OperationDesc;

/**
 * @brief The Operation class should describe the interface for performing signal processing on signal data.
 *
 * 'process' will only be called from one thread.
 */
class SaweDll Operation
{
public:
    typedef boost::shared_ptr<Operation> Ptr;
    typedef boost::weak_ptr<Operation> WeakPtr;


    /**
      Virtual housekeeping.
      */
    virtual ~Operation() {}


    /**
     * @brief process computes the operation
     * @param A buffer with data to process. The interval of the buffer will
     * be equal to param 'I' after a call to requiredInterval. requiredInterval
     * may be called several times for different intervals before calling
     * process.
     * @return processed data. Returned buffer interval must be equal to
     * OperationDesc::requiredInterval(b->getInterval());
     */
    virtual Signal::pBuffer process(Signal::pBuffer b) = 0;


    static void test(Ptr o, OperationDesc*);
};


/**
 * @brief The OperationDesc class should describe the interface for creating instances of the Operation interface.
 *
 * TODO should use VolatilePtr as the operation description might be accessed from different threads.
 */
class SaweDll OperationDesc
{
public:
    typedef boost::shared_ptr<OperationDesc> Ptr;


    /**
      Virtual housekeeping.
      */
    virtual ~OperationDesc() {}


    /**
     * @brief requiredInterval returns the interval that is required to compute
     * a valid chunk representing interval I. If the operation can not compute
     * a valid chunk representing the at least interval I at once the operation
     * can request a smaller chunk for processing instead by modifying I before
     * returning.
     * @param expectedOutput will overlap I. expectedOutput may be zero. If
     * 'expectedOutput' is non-zero it will be assigned an Interval by
     * requiredInterval.
     */
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const = 0;


    /**
     * @brief requiredInterval takes a Signal::Intervals with the same logic as
     * the version for Signal::Interval. The default implementation calls the
     * other version with I.fetchFirstInterval ();
     * Used to let an Operation prioritize in which order to compute stuff.
     * Such as starting computing closer to the camera.
     * @param 'I' may be modified by Operation if another interval is preferred.
     * I.first must be contained in a modified interval.
     */
    virtual Signal::Interval requiredInterval( const Signal::Intervals& I, Signal::Interval* expectedOutput ) const;


    /**
     * @brief affectedInterval
     * @param I
     * @return which interval of samples that needs to be recomputed if 'I'
     * changes in the input.
     */
    virtual Interval affectedInterval( const Interval& I ) const = 0;


    /**
     * @brief affectedInterval
     * @param I
     * @return a merge of the intervals returned by affectedInterval for each interval in I
     */
    virtual Intervals affectedInterval( const Intervals& I ) const;


    /**
     * @brief copy creates a copy of 'this'.
     * @return a copy.
     */
    virtual OperationDesc::Ptr copy() const = 0;


    /**
     * @brief createOperation instantiates an operation that uses this description.
     * Different computing engines could be used to instantiate different types.
     *
     * May return an empty Operation::Ptr if an operation isn't supported by a
     * given ComputingEngine in which case some other thread will have to
     * populate the cache instead and call invalidate samples when they are
     * ready. This thread will then have to wait, or do something else. It will
     * be marked as complete without computing anything until invalidated.
     * @return a newly created operation.
     */
    virtual Operation::Ptr createOperation(ComputingEngine* engine=0) const = 0;


    /**
     * @brief The SignalExtent struct is returned by OperationDesc::extent ()
     */
    struct Extent {
        boost::optional<Interval> interval;
        boost::optional<float> sample_rate;
        boost::optional<int> number_of_channels;
    };


    /**
     * @brief extent describes the extent of this operation. Extent::interval is allowed
     * to change during the lifetime of an OperationDesc.
     * @return Does not initialize anything unless this operation creates any special extent.
     */
    virtual Extent extent() const;


    /**
     * @brief recreateOperation recreates an operation in an existing instance.
     * If the operation supports this, some caches might be reused instead of
     * deallocated and reallocated (to reduce memory fragmentation). The
     * default behaviour is to call createOperation with the given engine.
     *
     * @return the same operation if it could be reused and modified, or a new
     * operation. The default is to not use the hint at all.
     *
     * TODO deprecated
     */
    virtual Operation::Ptr recreateOperation(Operation::Ptr hint, ComputingEngine* engine=0) const;


    /**
     * Returns a string representation of this operation. Mainly used for debugging.
     */
    virtual QString toString() const;


    /**
     * @brief getNumberOfSources is larger than 1 if the operation merges
     * multiple sources.
     *
     * TODO deprecated
     */
    virtual int getNumberOfSources() const;


    /**
     * @brief operator == checks if two instances of OperationDesc would generate
     * identical instances of Operation. The default behaviour is to just check
     * the type of the argument.
     * @param d
     * @return
     */
    virtual bool operator==(const OperationDesc& d) const;
    bool operator!=(const OperationDesc& b) const { return !(*this == b); }


    /**
     * @brief operator << makes OperationDesc support common printing routines
     * using the stream operator.
     * @param os
     * @param d
     * @return os
     */
    friend std::ostream& operator << (std::ostream& os, const OperationDesc& d);
};


/**
 * @brief The OperationSourceDesc class describes a starting point of a Dag.
 */
class SaweDll OperationSourceDesc: public OperationDesc
{
public:
    /**
     * @brief getNumberOfSources overloads OperationDesc::getNumberOfSources
     */
    int getNumberOfSources() const { return 0; }


    /**
     * @brief getSampleRate is constant during the lifetime of OperationSourceDesc.
     * @return the sample rate.
     */
    virtual float getSampleRate() const = 0;

    /**
     * @brief getNumberOfChannels is constant during the lifetime of OperationSourceDesc.
     * @return the number of channels.
     */
    virtual float getNumberOfChannels() const = 0;

    /**
     * @brief getNumberOfSamples does not have to be constant during the lifetime of OperationSourceDesc
     * @return the number of samples in this description.
     */
    virtual float getNumberOfSamples() const = 0;
};


/**
A Signal::Operation is a Signal::Source which reads data from another 
Signal::Source and performs some operation on that data before returning it to
the caller.
 */
class SaweDll DeprecatedOperation: public SourceBase
{
public:
    //typedef boost::shared_ptr<DeprecatedOperation> Ptr;

    /**
      This constructor by itself creates a dummy Operation that redirects 'read'
      to its _source.
      */
    DeprecatedOperation( pOperation source );
    ~DeprecatedOperation();

    DeprecatedOperation( const DeprecatedOperation& o );
    DeprecatedOperation& operator=(const DeprecatedOperation& o);

    /**
     * @brief name() human readable text description of *this
     * Returns a user friendly representation of this instance with a text
     * string to be displayed within the user interface (i.e in the window
     * with a list of Operations).
     * @return a human readable text string.
     */
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
    virtual float sample_rate() { return _source?_source->sample_rate():1.f; }
    virtual IntervalType number_of_samples(); /// @see read(const Interval&)
    virtual float length();

    virtual unsigned num_channels() { return _source?_source->num_channels():0; }


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
    virtual std::set<DeprecatedOperation*> outputs() { return _outputs; }

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
    virtual DeprecatedOperation* affecting_source( const Interval& I );


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

      translate_interval is used by Operation::zeroed_samples_recursive
      and Operation::invalidate_samples.

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


    DeprecatedOperation* root();
    virtual bool hasSource(DeprecatedOperation*s);

    /**
      If 's' has multiple outputs, find the output that leads to 'this' and
      return the parent of 's' from that trace.
      */
    static pOperation findParentOfSource(pOperation start, pOperation source);
    static Signal::Intervals affectedDiff(pOperation source1, pOperation source2);

    virtual std::string toString();
    virtual std::string toStringSkipSource();
    virtual std::string parentsToString();

private:
    std::set<DeprecatedOperation*> _outputs; /// @see Operation::parent()
    pOperation _source; /// @see Operation::source()
    //bool _enabled; /// @see Operation::enabled()

    friend class boost::serialization::access;
    DeprecatedOperation() /// only used by deserialization, call Operation(pOperation) instead
    {}
    template<class archive>
    void serialize(archive& ar, const unsigned int /*version*/)
    {
        TaskInfo ti("Serializing %s, source: %s",
                    name().c_str(), _source?_source->name().c_str():0);
        ti.tt().partlyDone();

        ar & BOOST_SERIALIZATION_NVP(_source);

        DeprecatedOperation::source(_source);
    }
};


/**
  FinalSource overloads Operation and disables support for Operation::source().
  It is in functionality almost equivalent to SourceBase but can be shipped
  around as pOperation.
  */
class FinalSource: public DeprecatedOperation
{
public:
    FinalSource() : DeprecatedOperation(pOperation()) {}

    virtual pBuffer read( const Interval& I ) = 0;
    virtual float sample_rate() = 0;
    virtual IntervalType number_of_samples() = 0;

    virtual unsigned num_channels() = 0;

    virtual Signal::Intervals zeroed_samples();

private:
    virtual pOperation source() const { return pOperation(); }
    virtual void source(pOperation)   { throw std::logic_error("FinalSource: Invalid call"); }
};

} // namespace Signal

#endif // SIGNALOPERATION_H
