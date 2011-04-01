#include "signal/operationcache.h"

static const bool D = false;

namespace Signal {

OperationCache::
        OperationCache( pOperation source )
:   Operation(source),
    _cache()
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
    Intervals cached = _cache.samplesDesc_current_channel() - _cache.invalid_samples_current_channel();

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
    _cache.validate_samples( zeroed - _cache.samplesDesc_current_channel() );
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

    unsigned c = get_channel();

    if (b)
    {
        Signal::Interval J = b->getInterval();

        _cache.put(b);

        if (1<b->channels())
        {
            Signal::Intervals toinv;
            for (unsigned d=0; d<b->channels(); d++)
            {
                if (_invalid_returns[d] & J)
                {
                    toinv |= _invalid_returns[d] & J;
                    _invalid_returns[d] -= J;
                }
            }
            Operation::invalidate_samples( toinv );
        }
        else if (_invalid_returns[c] & J)
        {
            Operation::invalidate_samples( _invalid_returns[c] & J );
            _invalid_returns[c] -= J;
        }

        cached = _cache.samplesDesc_current_channel() - _cache.invalid_samples_current_channel();
        ok = (cached & I).fetchFirstInterval();
        if (ok.first == I.first && ok.count())
        {
            return _cache.readFixedLength( ok );
        }

        missing = (I - cached).fetchFirstInterval();
    }

    _invalid_returns[c] |= missing;
    b = source()->readFixedLength( missing );
    _cache.put(b);
    return b;
}


void OperationCache::
        invalidate_samples(const Intervals& I)
{
    invalidate_cached_samples( I );
    Operation::invalidate_samples( I );
}


void OperationCache::
        invalidate_cached_samples(const Intervals& I)
{
    unsigned N = source() ? source()->num_channels() : num_channels();

    BOOST_ASSERT( 1 <= N );

    _invalid_returns.resize( N );
    _cache.setNumChannels( N );
    _cache.invalidate_samples( I );
    // TODO do this
    //_cache.invalidate_and_forget_samples(I);
}


Intervals OperationCache::
        invalid_samples()
{
    Intervals c = _cache.invalid_samples_all_channels();
    Interval i = getInterval();
    Intervals d = c & i;
    return d;
}


Intervals OperationCache::
        cached_samples()
{
    return _cache.samplesDesc_all_channels() - _cache.invalid_samples_all_channels() - invalid_returns();
}


Intervals OperationCache::
        invalid_returns()
{
    Intervals R;
    for (unsigned i=0; i<_invalid_returns.size(); ++i)
        R |= _invalid_returns[i];

    return R & getInterval();
}


unsigned OperationCache::
        num_channels()
{
    return _cache.num_channels();
}


void OperationCache::
        set_channel(unsigned c)
{
    _cache.set_channel( c );
    Operation::set_channel( c );
}


unsigned OperationCache::
        get_channel()
{
    return _cache.get_channel();
}


void OperationCache::
        source(pOperation v)
{
    if (v && dynamic_cast<Signal::FinalSource*>(v->root()))
    {
        unsigned N = v->num_channels();
        _invalid_returns.resize( N );
        _cache.setNumChannels( N );
    }

    Operation::source( v );
}


OperationCachedSub::
        OperationCachedSub( pOperation source )
    :
    OperationCache(source)
{
    BOOST_ASSERT( source );
}


std::string OperationCachedSub::
        name()
{
    return Operation::source()->name();
}


Signal::Intervals OperationCachedSub::
        affected_samples()
{
    return Operation::source()->affected_samples();
}


pBuffer OperationCachedSub::readRaw( const Interval& I )
{
    return Operation::read(I);
}


void OperationCachedSub::
        source(pOperation v)
{
    pOperation o = Operation::source();
    o->source( v );
    OperationCache::source( o );
}


pOperation OperationCachedSub::
        source()
{
    return Operation::source()->source();
}


} // namespace Signal
