#ifndef SIGNALOPERATIONCACHE_H
#define SIGNALOPERATIONCACHE_H

#include "signal/operation.h"
#include "signal/sinksourcechannels.h"

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

    virtual void invalidate_samples(const Intervals& I);

    virtual Intervals invalid_samples();
    virtual Intervals invalid_returns();

    /**
      Function to read from on a cache miss. Doesn't have to return the data
      that was actually requested for. It may also return null if the data is
      not ready yet.

      If readRaw ever returns null there must be a watchdog somewhere to
      ensure that invalidate_samples(I) are called when data is made available.
      */
    virtual pBuffer readRaw( const Interval& I ) = 0;

    virtual unsigned num_channels();
    virtual void set_channel(unsigned c);
    virtual unsigned get_channel();

    virtual void source(pOperation v);
    virtual pOperation source() { return Operation::source(); }

protected:
    SinkSourceChannels _cache;

private:
    /**
      OperationCache populates this when readRaw doesn't return the expected interval.
      It is up to an implementation to use this information somehow, for
      instance by issuing Operation::invalidate_samples(). To notify callers
      that the information is now available.
      */
    std::vector<Signal::Intervals> _invalid_returns;
};


class OperationCacheLayer: public OperationCache
{
public:
    OperationCacheLayer( pOperation source ):OperationCache(source){}
    virtual Signal::Intervals affected_samples() { return source()->affected_samples(); }
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
