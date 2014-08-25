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
      Extract an exact interval from cache. Samples in
      "I - sampleDesc()" will be returned as zeros.
      */
    pBuffer read( const Interval& I ) const;

    /**
      Extract an exact interval from cache. Samples in
      "r->getInterval() - sampleDesc()" will be left as is.
      */
    void read( pBuffer r ) const;

    /**
      A slightly more efficient version than read(I) that is only guaranteed to
      return a buffer containing I.first. On a cache miss this method returns a
      buffer with zeros of the requested interval 'I' or smaller.
     */
    pBuffer readAtLeastFirstSample( const Interval&I ) const;

    /// Clear cache, also clears invalid_samples
    void clear();

    /**
      Insert data into Cache, may throw InvalidBufferDimensions.
      @throw InvalidBufferDimensions if 'fs' and/or 'num_channels' of 'b'
      does not match what has already been allocated.
      */
    void put( pBuffer b );

    /// Get what samples that are described in the containing buffer
    /// Merely allocated memory does not count in this description.
    Intervals samplesDesc() const;
    Interval spannedInterval() const;
    Intervals allocated() const;
    /**
     * @brief contains checks if all of I is covered by the cache
     * @return samplesDesc().contains(I)
     */
    bool contains(const Signal::Intervals& I) const;
    bool empty() const;

    void invalidate_samples(const Intervals& I);
    Signal::Intervals purge(Signal::Intervals still_needed);
    size_t cache_size() const;

    /**
     * @brief num_channels is defined as 0 if _cache is empty.
     * @return
     */
    int num_channels() const;

private:
    std::vector<pBuffer> _cache;
    std::vector<pBuffer> _discarded;

    /**
     * @brief _valid_samples explains the samples that can be fetched from
     * this instance. Trying to read anything outside of this will yield an
     * empty buffer with zeroes.
     */
    Intervals _valid_samples;

    /**
     * @brief allocateCache
     * @param fs
     * @param num_channels
     * @throw InvalidBufferDimensions if fs and/or num_channels doesn't match
     * what's already been allocated.
     */
    void allocateCache( const Signal::Interval&, float fs, int num_channels );

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
