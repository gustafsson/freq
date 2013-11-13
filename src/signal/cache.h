#ifndef SIGNAL_CACHE_H
#define SIGNAL_CACHE_H

#include "buffer.h"

#include <boost/exception/all.hpp>

#include <vector>

namespace Signal {


/**
 * @brief The Cache class
 * Not thread-safe.
 */
class Cache
{
public:
    class InvalidBufferDimensions: virtual public boost::exception, virtual public std::exception {};

    Cache( );
    Cache( const Cache& b);
    Cache& operator=( const Cache& b);

    /**
      'sample_rate' is defined as 0 if _cache is empty.
      If buffers with different are attempted to be 'put' then 'put' will throw
      an invalid_argument exception. So all buffers in the cache are guaranteed
      to have the same sample rate as the buffer that was first inserted with
      'put'.
      */
    float sample_rate() const;

    /**
      Extract an exact interval from cache. Samples in the interval
      "I - sampleDesc()" will be returned as zeros.
      */
    pBuffer read( const Interval& I ) const;

    /**
      A slightly more efficient version than read(I) that is only guaranteed to
      return a buffer containing I.first. On a cache miss this method returns a
      buffer with zeros of the requested interval 'I' or smaller.
     */
    pBuffer readAtLeastFirstSample( const Interval&I ) const;

    /// Clear cache, also clears invalid_samples
    void clear();

    /**
      Insert data into Cache
      */
    void put( pBuffer b );

    /// Get what samples that are described in the containing buffer
    /// Merely allocated memory doesn't not count in this description.
    Intervals samplesDesc() const;

    void invalidate_samples(const Intervals& I);

    /// Return true if the entire interval I is up to date and can be read from this.
    bool hasInterval(const Interval& I);

    int num_channels() const;

private:
    std::vector<pBuffer> _cache;

    /**
     * @brief _valid_samples explains the samples that can be fetched from
     * this instance. Trying to read anything outside of this will yield an
     * empty buffer with zeroes.
     */
    Intervals _valid_samples;

    void allocateCache( Signal::Interval, float fs, int num_channels );
    void merge( pBuffer );

    /**
     * @brief findBuffer finds the buffer containing 'sample'.
     * Assumes that the _cache_mutex is locked already.
     * @return The buffer containing 'sample' or the next following buffer if
     * no buffer contains 'sample'. Might be _cache.end()
     */
    std::vector<pBuffer>::iterator findBuffer( Signal::IntervalType sample );
    std::vector<pBuffer>::const_iterator findBuffer( Signal::IntervalType sample ) const;

public:
    static void test();
};

} // namespace Signal

#endif // SIGNAL_CACHE_H
