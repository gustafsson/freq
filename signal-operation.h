#ifndef SIGNALOPERATION_H
#define SIGNALOPERATION_H

#include "signal-source.h"
#include "signal-samplesintervaldescriptor.h"

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
      This constructor by itself creates a dummy Operation that redirects any
      method calls to its _source.
      */
    Operation( pSource source );

    /**
      The default implementation of read is to read from source()
      */
    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples )
    {
        return source()->read(firstSample, numberOfSamples);
    }

    virtual pBuffer readChecked( unsigned firstSample, unsigned numberOfSamples );
    virtual pBuffer readFixedLength( unsigned firstSample, unsigned numberOfSamples );

    virtual unsigned sample_rate();
    virtual unsigned number_of_samples();
    virtual pSource source() const { return _source; }
    virtual void source(pSource v) { _source=v; }

    virtual SamplesIntervalDescriptor updateInvalidSamples();

    static pSource first_source(pSource start);

    /// finds last source that does not have a slow operation (i.e. FilterOperation) among its parents.
    static pSource fast_source(pSource start);

protected:    
    // TODO define how _invalid_samples is used
    SamplesIntervalDescriptor _invalid_samples;
    pSource _source;
};

} // namespace Signal

#endif // SIGNALOPERATION_H
