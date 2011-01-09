#ifndef SIGNALSINKSOURCE_H
#define SIGNALSINKSOURCE_H

#include "signal/sink.h"
#include "signal/source.h"
#include "signal/intervals.h"
#include <vector>
#ifndef SAWE_NO_MUTEX
#include <QMutex>
#endif

namespace Signal {

/**
  Both a sink and a source at the same time. Well, yes. Like an iostream.

  You insert data with 'SinkSource::put(pBuffer)', it will be located where the
  buffer sais the data is. If the SinkSource already contains data for that
  location it will be overwritten.

  There are two different accept strategies that sais whether put should deny
  parts of incomming buffers or not. Expected samples are given by
  'Operation::invalid_samples()'.

  Afterwards, any data put into the SinkSource can be fetched with
  'SinkSource::read'. If the read Interval starts at a location where no data
  has been put zeros will be returned for that section. Otherwise 'read' will
  return a block from its internal cache that includes 'Interal::first' but
  could otherwise be both much larger and much smaller than the requested
  length. Use 'Source::readFixedLength' if you need specific samples.
  */
class SinkSource: public Sink
{
public:
    /// @see SinkSource
    SinkSource();
	SinkSource( const SinkSource& b);
	SinkSource& operator=( const SinkSource& b);

    /**
      Insert data into SinkSource
      */
    void put( pBuffer b );


    /**
      Samples in 'b' will only be accepted if they are present in 'expected'.
      */
    void putExpectedSamples( pBuffer b, const Intervals& expected );


    /// Clear cache
    void clear();

    /**
      Extract an interval from cache, only guaranteed to return a buffer
      containing I.first. On a cache miss this method returns a buffer with
      zeros, of the requested interval 'I' or smaller.
      */
    virtual pBuffer read( const Interval& I );

    /**
      'sample_rate' is defined as 0 if _cache is empty.
      If buffers with different are attempted to be 'put' then 'put' will throw
      an invalid_argument exception. So all buffers in the cache are guaranteed
      to have the same sample rate as the buffer that was first inserted with
      'put'.
      */
    virtual float sample_rate();

    /**
      Total number of sampels in cached interval, equal to
        'samplesDesc().coveredInterval().count'.
      */
    virtual long unsigned number_of_samples();

    /// The first buffer in the cache, or pBuffer() if cache is empty
    pBuffer first_buffer();

    /// If cache is empty
    bool empty();

    /// Get what samples that are described in the containing buffer
    Intervals samplesDesc();

    /// @see Operation::fetch_invalid_samples()
    //virtual void invalidate_samples(const Intervals& I) { _invalid_samples |= samplesDesc()&I; }
    virtual void invalidate_samples(const Intervals& I) { _invalid_samples |= I; }

private:
#ifndef SAWE_NO_MUTEX
	QMutex _cache_mutex;
#endif
    std::vector<pBuffer> _cache; // todo use set instead

    virtual pOperation source() const { return pOperation(); }
    virtual void source(pOperation)   { throw std::logic_error("Invalid call"); }

    void selfmerge();
    void merge( pBuffer );
};

typedef boost::shared_ptr<Sink> pSink;

} // namespace Signal

#endif // SIGNALSINKSOURCE_H
