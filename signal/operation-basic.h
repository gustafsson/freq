#ifndef SIGNALOPERATIONBASIC_H
#define SIGNALOPERATIONBASIC_H

#include "signal/operation.h"

namespace Signal {

class OperationRemoveSection: public Operation
{
public:
    OperationRemoveSection( pSource source, unsigned firstSample, unsigned numberOfRemovedSamples );

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );
    virtual long unsigned number_of_samples();
private:
    unsigned _firstSample, _numberOfRemovedSamples;
};

class OperationInsertSilence: public Operation
{
public:
    OperationInsertSilence( pSource source, unsigned firstSample, unsigned numberOfSilentSamples );

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );
    virtual long unsigned number_of_samples();
private:
    unsigned _firstSample, _numberOfSilentSamples;
};

class OperationSuperposition: public Operation
{
public:
    OperationSuperposition( pSource source, pSource source2 );

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );

    virtual pSource source2() const { return _source2; }

private:
    pSource _source2;
};

} // namespace Signal

#endif // SIGNALOPERATIONBASIC_H
