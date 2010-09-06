#ifndef SIGNALOPERATION_H
#define SIGNALOPERATION_H

#include "signal/source.h"
#include "signal/intervals.h"

namespace Signal {

/**
A Signal::Operation is a Signal::Source which reads data from another 
Signal::Source and performs some operation on that data before returning it to
the caller.
 */
class Operation: public Source
{
public:
    /**
      This constructor by itself creates a dummy Operation that redirects 'read'
      to its _source.
      */
    Operation( pSource source );

    /// The default implementation of read is to read from source()
    virtual pBuffer read( Interval I )
    {
        return source()->read( I );
    }

    /**
      If this Operation acts as a passive operation multiple Operations that
      would otherwise run in parallell can be chained into eachother. An
      operation is allowed to be passive in some parts and nonpassive in others.
      'nonpassive_operation' describes where. 'nonpassive_operation' is allowed
      to change over time as well, but invalid_samples must be changed
      accordingly for new intervals to be computed.
      */
    virtual Intervals nonpassive_operation() { return Intervals(); }


    virtual unsigned sample_rate();
    virtual long unsigned number_of_samples();
    virtual pSource source() const { return _source; }
    virtual void source(pSource v) { _source=v; }

    /// Returns _invalid_samples merged with _source->invalid_samples.
    virtual Intervals invalid_samples();

    static pSource first_source(pSource start);

    /// finds last source that does not have a slow operation (i.e. CwtFilter) among its sources.
    static pSource fast_source(pSource start);

    /// Finds
    static pSource non_place_holder(pSource start);

protected:
    pSource _source;

    /**
      _invalid_samples describes which samples that should be re-read off from
      Operation. It is initialized to an empty interval and can be
      used by an implementaion to say that the previous results are out of
      date. If an implementaion use this feature it must also gradually work
      it off by calls to validate_samples.
      */
    Intervals _invalid_samples;
};

} // namespace Signal

#endif // SIGNALOPERATION_H
