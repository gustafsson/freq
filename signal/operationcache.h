#ifndef SIGNALFILTERCACHE_H
#define SIGNALFILTERCACHE_H

#include "signal/operation.h"
#include "signal/sinksource.h"

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

    /**
      Redirects the read to '_cache' unless cacheMiss returns true in which
      case it reads from 'readRaw'.
      */
    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );

    /**
      Function to read from on a cache miss
      */
    virtual pBuffer readRaw(unsigned firstSample, unsigned numberOfSamples ) = 0;

    /**
      Defines what a cache miss is, default implementation checks if the entire
      sample range exists in _data. If any sample is non-existent in _data it is
      a cache miss.
      */
    virtual bool cacheMiss(unsigned firstSample, unsigned numberOfSamples);

private:
    SinkSource _cache;
};

} // namespace Signal

#endif // SIGNALFILTERCACHE_H
