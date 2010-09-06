#include "postsink.h"
#include "cwtfilter.h"

#include <boost/foreach.hpp>

namespace Signal {

void PostSink::
        put( pBuffer b, pSource s)
{
    Signal::Intervals expected = expected_samples();
    if ( (expected & b->getInterval()).isEmpty() )
        return;  // Don't forward data that wasn't requested

    if (_inverse_cwt.filter.get())
    {
        // Get a chunk for this block
        Tfr::pChunk chunk = getChunk(b,s);

        if ( (expected & chunk->getInterval()).isEmpty() )
            return; // Don't forward data that wasn't requested

        // Discard given buffer
        b.reset();

        b = _inverse_cwt( *chunk );
    }

    BOOST_FOREACH( pSink sink, sinks() )
        sink->put(b, s);
}

void PostSink::
        reset()
{
    BOOST_FOREACH( pSink s, sinks() )
        s->reset( );

    Tfr::ChunkSink::reset();
}

bool PostSink::
        isFinished()
{
    bool r = true;

    BOOST_FOREACH( pSink s, sinks() )
        r &= s->isFinished( );

    return r;
}

void PostSink::
        onFinished()
{
    BOOST_FOREACH( pSink s, sinks() )
        s->onFinished( );
}

Intervals PostSink::
        expected_samples()
{
    Intervals x;

    BOOST_FOREACH( pSink s, sinks() )
        x |= s->expected_samples();

    _expected_samples = x;
    return x;
}

void PostSink::
        add_expected_samples( const Intervals& x )
{
    BOOST_FOREACH( pSink s, sinks() )
        s->add_expected_samples( x );

    _expected_samples |= x;
}

std::vector<pSink> PostSink::
        sinks()
{
    QMutexLocker l(&_sinks_lock);
    return _sinks;
}

void PostSink::
        sinks(std::vector<pSink> v)
{
    QMutexLocker l(&_sinks_lock);
    _sinks = v;
}

Tfr::pFilter PostSink::
        filter()
{
    return _inverse_cwt.filter;
}

void PostSink::
        filter(Tfr::pFilter f, pSource s)
{
    if (f!=_inverse_cwt.filter) {
        sinks(std::vector<pSink>());
    }

    reset();

    if (f) {
        unsigned FS = s->sample_rate();
        Interval i =
                (f->getTouchedSamples(FS) - f->ZeroedSamples(FS)).coveredInterval();

        i.last = std::min( i.last, (Intervals::SampleType)s->number_of_samples() );

        if (!i.valid())
            f.reset();
        else
            add_expected_samples( i );

    }

    // Change filter
    _inverse_cwt.filter = f;
}

} // namespace Signal
