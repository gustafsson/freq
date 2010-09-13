#ifndef SIGNALOPERATION_H
#define SIGNALOPERATION_H

#include "signal/source.h"
#include "signal/intervals.h"
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/shared_ptr.hpp>

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
      from source()
      */
    virtual pBuffer read( const Interval& I ) { return source()->read( I ); }
    virtual unsigned sample_rate() { return source()->sample_rate(); }  /// @see read(Interval)
    virtual long unsigned number_of_samples() { return source()->number_of_samples(); } /// @see read(Interval)


    /**
      What makes an operation is that it processes a signal that actually comes
      from somewhere else.
      */
    virtual pOperation source() const { return _source; }
    virtual void source(pOperation v) { _source=v; } /// @see source()


    /**
      In short AffectedSamples describes where it is possible that
        'source()->read( I ) != read( I )'

      If this Operation acts as a passive operation then multiple Operations
      that would otherwise run in parallell can be chained into eachother. An
      operation is allowed to be passive in some parts and nonpassive in others.
      'AffectedSamples' describes where. 'AffectedSamples' is allowed to change
      over time as well, but invalid_samples must be changed accordingly for new
      intervals to be computed.
      */
    virtual Intervals affected_samples() { return Intervals(); }

    /// Returns _invalid_samples merged with source()->invalid_samples().
    virtual Intervals invalid_samples();
    /// @see _invalid_samples
    virtual void invalidate_samples(const Intervals& I) { _invalid_samples |= I; }



    static pOperation first_source(pOperation start);

    /// finds last source that does not have a slow operation (i.e. CwtFilter) among its sources.
    static pOperation fast_source(pOperation start);

    /// Finds
    // todo remove static pOperation non_place_holder(pOperation start);

protected:
    pOperation _source;

    /**
      _invalid_samples describes which samples that should be re-read off from
      Operation. It is initialized to an empty interval and can be
      used by an implementaion to say that the previous results are out of
      date. If an implementaion use this feature it must also gradually work
      it off in calls to read.
      */
    Intervals _invalid_samples;

private:
    Operation() {} // used by serialization
    friend class boost::serialization::access;
    template<class archive>
    void serialize(archive& ar, const unsigned int /*version*/)
    {
        using boost::serialization::make_nvp;
        ar & make_nvp("Source", _source);
    }
};


class FinalSource: public Operation
{
public:
    FinalSource():Operation(pOperation()) {}

    virtual pBuffer read( const Interval& I ) = 0;
    virtual unsigned sample_rate() = 0;
    virtual long unsigned number_of_samples() = 0;

private:
    virtual pOperation source() const { return pOperation(); }
    virtual void source(pOperation)   { throw std::logic_error("Invalid call"); }
};

} // namespace Signal

#endif // SIGNALOPERATION_H
