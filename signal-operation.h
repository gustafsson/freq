#ifndef SIGNALOPERATION_H
#define SIGNALOPERATION_H

#include "signal-source.h"
#include "signal-invalidsamplesdescriptor.h

namespace Signal {

class Operation: public Source
{
public:
    Operation( pSource child );
    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples ) = 0;
    virtual unsigned sample_rate() const;
    virtual unsigned number_of_samples() const;

    virtual InvalidSamplesDescriptor updateIsd();

protected:
    InvalidSamplesDescriptor _isd;
    pSource _child;
};

} // namespace Signal

#endif // SIGNALOPERATION_H
