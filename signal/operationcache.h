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

private:
    friend class boost::serialization::access;
    OperationCacheLayer() : OperationCache(pOperation()) {}
    template<class Archive> void serialize(Archive& ar, const unsigned int /*version*/) {
        TaskInfo("OperationCacheLayer::serialize");
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
    }
};

} // namespace Signal

#endif // SIGNALOPERATIONCACHE_H
