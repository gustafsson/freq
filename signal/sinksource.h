#ifndef SIGNALSINKSOURCE_H
#define SIGNALSINKSOURCE_H

#include "signal/sink.h"
#include "signal/source.h"
#include "signal/intervals.h"
#include <vector>
#include <QMutex>

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
class SinkSource: public Operation
{
public:
    /// @see put
    enum AcceptStrategy {
        AcceptStrategy_ACCEPT_ALL,
        AcceptStrategy_ACCEPT_EXPECTED_ONLY
    };

    /// @see SinkSource
    SinkSource( AcceptStrategy a );

    /**
      Insert data into SinkSource, if the AcceptStrategy is 'excepted only'
      samples will only be accepted if they are first announced with
      'invalidate_samples', i.e. 'invalidate_samples(b->getInterval())'.
      */
    void put( pBuffer b );

    /// Clear cache
    void reset();

    /**
      Extract an interval from cache, only guaranteed to return a buffer
      containung I.first.
      */
    virtual pBuffer read( const Interval& I );

    /**
      'sample_rate' is defined as (unsigned)-1 if _cache is empty.
      If buffers with different are attempted to be 'put' then 'put' will throw
      an invalid_argument exception. So all buffers in the cache are guaranteed
      to have the same sample rate as the buffer that was first inserted with
      'put'.
      */
    virtual unsigned sample_rate();

    /**
      Total number of sampels cached. Use samplesDesc.coveredInterval().count
      */
    // virtual long unsigned number_of_samples();

    /// The first buffer in the cache, or pBuffer() if cache is empty
    pBuffer first_buffer();

    /// If cache is empty
    bool empty();

    /// Get what samples that are described in the containing buffer
    Intervals samplesDesc();

private:
    QMutex _cache_mutex;
    std::vector<pBuffer> _cache;
    AcceptStrategy _acceptStrategy;

    void selfmerge();
    void merge( pBuffer );
};

typedef boost::shared_ptr<Sink> pSink;

} // namespace Signal

#endif // SIGNALSINKSOURCE_H
