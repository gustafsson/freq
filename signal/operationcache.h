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

    virtual void invalidate_samples(const Intervals& I) { _cache.invalidate_samples(I); Operation::invalidate_samples(I); }

    virtual Intervals invalid_samples();
    virtual Intervals invalid_returns();

    /**
      Function to read from on a cache miss
      */
    virtual pBuffer readRaw( const Interval& I ) = 0;

protected:
    SinkSource _cache;

    /**
      OperationCache populates this when readRaw doesn't return the expected interval.
      It is up to an implementation to use this information somehow, for
      instance by issueing Operation::invalidate_samples(). To notify callers
      that the information is now available.
      */
    Signal::Intervals _invalid_returns;
};

class OperationCacheLayer: public OperationCache
{
public:
    OperationCacheLayer( pOperation source ):OperationCache(source){}
    virtual Signal::Intervals affected_samples() { return Signal::Intervals(); }
    virtual pBuffer readRaw( const Interval& I ) { return Operation::read(I); }

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
