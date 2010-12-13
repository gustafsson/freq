#ifndef SIGNALOPERATIONCACHE_H
#define SIGNALOPERATIONCACHE_H

#include "signal/operation.h"
#include "signal/sinksource.h"

namespace Signal {


/**
  OperationCache reads from its source if the requested samples have not been
  previously requested or if the samples are not in invalid_samples().

  Otherwise, cached result from previous reads are immediately
  returned.
  */
class OperationCache: public Operation
{
public:
    OperationCache( pOperation source );

    /**
      Redirects the read to '_cache' unless cacheMiss returns true in which
      case it reads from 'readRaw'.
      */
    virtual pBuffer read( const Interval& I );

    /**
      Function to read from on a cache miss
      */
    virtual pBuffer readRaw( const Interval& I ) = 0;

    /**
      Defines what a cache miss is, default implementation checks if the entire
      sample range exists in _data. If any sample is non-existent in _data it is
      a cache miss.
      */
    virtual bool cacheMiss( const Interval& I );

protected:
    SinkSource _cache;
};

class OperationCacheLayer: public OperationCache
{
public:
    OperationCacheLayer( pOperation source ):OperationCache(source){}
    virtual pBuffer readRaw( const Interval& I ) { return Operation::read(I); }
    virtual void invalidate_samples(const Intervals& I) { _cache.invalidate_samples(I); }
    virtual Intervals fetch_invalid_samples() { return _cache.fetch_invalid_samples() | Operation::fetch_invalid_samples(); }
};

} // namespace Signal

#endif // SIGNALOPERATIONCACHE_H
