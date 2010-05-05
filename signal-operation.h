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
    Operation( pSource source );

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples ) = 0;

    virtual unsigned sample_rate();
    virtual unsigned number_of_samples();
    virtual pSource source() const { return _source; }

    virtual SamplesIntervalDescriptor updateInvalidSamples();

    // TODO should find last source that does not have a slow operation (i.e. FilterOperation) among its parents.
    static pSource first_source(pSource start);

protected:
    SamplesIntervalDescriptor _invalid_samples;
    pSource _source;
};

} // namespace Signal

#endif // SIGNALOPERATION_H
