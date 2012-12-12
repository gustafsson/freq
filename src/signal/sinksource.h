#ifndef SIGNALSINKSOURCE_H
#define SIGNALSINKSOURCE_H

#include "signal/sink.h"
#include "signal/source.h"
#include "signal/intervals.h"
#include <vector>

#ifndef SAWE_NO_SINKSOURCE_MUTEX
#include <QMutex>
#endif

namespace Signal {

/**
  Both a sink and a source at the same time. Well, yes. Like an iostream.

  You insert data with 'SinkSource::put(pBuffer)', it will be located where the
  buffer sais the data is. If the SinkSource already contains data for that
  location it will be overwritten.

  There are two different accept strategies that sais whether put() should deny
  parts of incomming buffers or not. put() accepts anything while
  putExpectedSamples() discards all data that has not previously been marked as
  exptected by invalidate_samples. Expected samples are given by
  invalid_samples().

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
    typedef boost::shared_ptr<SinkSource> Ptr;

    /// @see SinkSource
    SinkSource( int num_channels );
    SinkSource( const SinkSource& b);
    SinkSource& operator=( const SinkSource& b);

    /**
      Insert data into SinkSource
      */
    void put( pBuffer b );

    /**
      Samples in 'b' will only be accepted if they are present in 'invalid_samples'.
      */
    void putExpectedSamples( pBuffer b )
    {
        putExpectedSamples( b, invalid_samples() );
    }

    virtual Intervals invalid_samples() { return _invalid_samples; }
    virtual void invalidate_samples(const Intervals& I);
    void invalidate_and_forget_samples(const Intervals& I);
    void validate_samples( const Intervals& I ) { _invalid_samples -= I; }

    /// Clear cache, also clears invalid_samples
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
        'samplesDesc().spannedInterval().count'.
      */
    virtual IntervalType number_of_samples();
    virtual unsigned num_channels() { return _num_channels; }


    /**
      First and last sample in the spanned interval, or [0,0).
      */
    virtual Interval getInterval();

    /// The first buffer in the cache, or pBuffer() if cache is empty
    pBuffer first_buffer();

    /// If cache is empty
    bool empty();

    /// Get what samples that are described in the containing buffer
    Intervals samplesDesc();

    /// Return true if the entire interval I is up to date and can be read from this.
    bool hasInterval(const Interval& I);

private:
#ifndef SAWE_NO_SINKSOURCE_MUTEX
	QMutex _cache_mutex;
#endif
    std::vector<pBuffer> _cache;

    /**
      Samples in 'b' will only be accepted if they are present in 'expected'.
      */
    void putExpectedSamples( pBuffer b, const Intervals& expected );

    /**
      _expected_samples describes which samples that should be put into this
      SinkSource. It is initialized to an empty interval and can be used through
      invalidate_samples() to say that certain samples are missing before
      calling putExpectedSamples.

      @see SinkSource::invalid_samples(), SinkSource::invalidate_samples()
      */
    Intervals _invalid_samples;

    /**
     * @brief _valid_samples explains the samples that can be fetched from
     * this instance. Trying to read anything outside of this will yield an
     * empty buffer with zeroes.
     */
    Intervals _valid_samples;

    int _num_channels;

    virtual pOperation source() const { return pOperation(); }
    virtual void source(pOperation)   { throw std::logic_error("Invalid call"); }

    void allocateCache( Signal::Interval, float fs );
    //void selfmerge( Signal::Intervals forget = Signal::Intervals() );
    void merge( pBuffer );

    /**
     * @brief findBuffer finds the buffer containing 'sample'.
     * Assumes that the _cache_mutex is locked already.
     * @return The buffer containing 'sample' or the next following buffer if
     * no buffer contains 'sample'. Might be _cache.end()
     */
    std::vector<pBuffer>::iterator findBuffer( Signal::IntervalType sample );
};

typedef SinkSource::Ptr pSinkSource;

} // namespace Signal

#endif // SIGNALSINKSOURCE_H
