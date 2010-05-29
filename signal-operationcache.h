#ifndef SIGNALFILTERCACHE_H
#define SIGNALFILTERCACHE_H

#include "signal-operation.h"
#include "signal-sinksource.h"

namespace Signal {

/**
  FilterCache is a dummy operation if source is not a FilterOperation.

  If source is a FilterOperation, Cache only read from FilterOperation if the
  requested samples have not been previously requested or if the samples
  are not in FilterOperation::invalid_samples().

  Otherwise, cached result from previous FilterOperation reads are immediately
  returned.
  */
class OperationCache: public Operation
{
public:
    OperationCache( pSource source );

    virtual pBuffer readRaw(unsigned firstSample, unsigned numberOfSamples ) = 0;

private:
    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );

    SinkSource _data;
};

} // namespace Signal

#endif // SIGNALFILTERCACHE_H
