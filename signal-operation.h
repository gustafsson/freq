#ifndef SIGNALOPERATION_H
#define SIGNALOPERATION_H

#include "signal-source.h"
#include "signal-samplesintervaldescriptor.h"

namespace Signal {

class Operation: public Source
{
public:
    Operation( pSource source );

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples ) = 0;

    virtual unsigned sample_rate();
    virtual unsigned number_of_samples();
    virtual pSource source() const { return _source; }

    virtual SamplesIntervalDescriptor updateInvalidSamples();

protected:
    SamplesIntervalDescriptor _invalid_samples;
    pSource _source;
};

} // namespace Signal

#endif // SIGNALOPERATION_H
