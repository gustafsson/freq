#include "signal/operationcache.h"

static const bool D = false;

namespace Signal {

OperationCache::
        OperationCache( pOperation source )
:   DeprecatedOperation(source),
    _cache(source?source->num_channels ():0)
{
    this->source(source);
}


pBuffer OperationCache::
        read( const Interval& I )
{
    static bool enable_cache = true;

    Interval ok = (~affected_samples() & I).fetchFirstInterval();

    // TODO do this
    /*if (ok.first == I.first && ok.count() && enable_cache)
    {
        // Wouldn't affect these samples, don't bother caching the results
        pBuffer b = source()->readFixedLength( ok );
        if (D) TaskTimer("%s: cache not affecting [%u, %u] got [%u, %u]",
                     __FUNCTION__,
                     I.first,
                     I.last,
                     b->getInterval().first,
                     b->getInterval().last).suppressTiming();
        return b;
    }*/


    // cached samples doesn't count in samplesDesc if they are marked as invalid
    Intervals cached = _cache.samplesDesc() - _cache.invalid_samples();

    ok = (cached & I).fetchFirstInterval();

    if (ok.first == I.first && ok.count() && enable_cache)
    {
        // Don't need anything new, return cache
        pBuffer b = _cache.readFixedLength( ok );
        if (D) TaskTimer("%s: cache [%u, %u] got [%u, %u]",
                     __FUNCTION__,
                     I.first,
                     I.last,
                     b->getInterval().first,
                     b->getInterval().last).suppressTiming();
        return b;
    }

    Intervals zeroed = zeroed_samples_recursive();
    _cache.validate_samples( zeroed - _cache.samplesDesc() );
    Interval first_zero = (I & zeroed).fetchFirstInterval();
    Interval missing = (I - cached - zeroed).fetchFirstInterval();

    pBuffer b;
    if (first_zero.first == I.first && first_zero.count())
        b = zeros(first_zero);
    else
        b = readRaw( missing );

    if (D) TaskTimer tt("%s: raw [%u, %u] got [%u, %u]",
                 __FUNCTION__,
                 I.first,
                 I.last,
                 b?b->getInterval().first:0,
                 b?b->getInterval().last:0);

    if (b)
    {
        Signal::Interval J = b->getInterval();

        _cache.put(b);

        if (_invalid_returns & J)
        {
            DeprecatedOperation::invalidate_samples( _invalid_returns & J );
            _invalid_returns -= J;
        }

        cached = _cache.samplesDesc () - _cache.invalid_samples ();
        ok = (cached & I).fetchFirstInterval();
        if (ok.first == I.first && ok.count())
        {
            return _cache.readFixedLength( ok );
        }

        missing = (I - cached).fetchFirstInterval();
    }

    _invalid_returns |= missing;

    if (source())
        b = source()->readFixedLength( missing );
    else
        b = zeros( missing );

    _cache.put(b);
    return b;
}


void OperationCache::
        invalidate_samples(const Intervals& I)
{
    invalidate_cached_samples( I );
    DeprecatedOperation::invalidate_samples( I );
}


void OperationCache::
        invalidate_cached_samples(const Intervals& I)
{
    unsigned N = source() ? source()->num_channels() : num_channels();

    EXCEPTION_ASSERT( 1 <= N );

    if (N != _cache.num_channels ())
        _cache = SinkSource(N);
    _cache.invalidate_samples( I );
    // TODO do this
    //_cache.invalidate_and_forget_samples(I);
}


Intervals OperationCache::
        invalid_samples()
{
    Intervals c = _cache.invalid_samples();
    Interval i = getInterval();
    Intervals d = c & i;
    return d;
}


Intervals OperationCache::
        cached_samples()
{
    return _cache.samplesDesc() - _cache.invalid_samples() - invalid_returns();
}


Intervals OperationCache::
        invalid_returns()
{
    return _invalid_returns & getInterval();
}


unsigned OperationCache::
        num_channels()
{
    return _cache.num_channels();
}


void OperationCache::
        source(pOperation v)
{
    if (v && dynamic_cast<Signal::FinalSource*>(v->root()))
    {
        unsigned N = v->num_channels();
        _cache = SinkSource(N);
    }

    DeprecatedOperation::source( v );
}


OperationCachedSub::
        OperationCachedSub( pOperation source )
    :
    OperationCache(source)
{
    EXCEPTION_ASSERT( source );
}


std::string OperationCachedSub::
        name()
{
    return (DeprecatedOperation::source()?DeprecatedOperation::source()->name():"(null)");
}


Signal::Intervals OperationCachedSub::
        affected_samples()
{
    return DeprecatedOperation::source()->affected_samples();
}


pBuffer OperationCachedSub::readRaw( const Interval& I )
{
    return DeprecatedOperation::read(I);
}


void OperationCachedSub::
        source(pOperation v)
{
    pOperation o = DeprecatedOperation::source();
    o->source( v );
    OperationCache::source( o );
}


pOperation OperationCachedSub::
        source() const
{
    return DeprecatedOperation::source()->source();
}


} // namespace Signal
