#include "signal-postsink.h"
#include "signal-filteroperation.h"
#include <boost/foreach.hpp>

namespace Signal {

void PostSink::
        put( pBuffer b, pSource s)
{
    TaskTimer tt("PostSink::put");

    if (inverse_cwt.filter.get())
    {
        // Get a chunk for this block
        Tfr::pChunk chunk = getChunk(b,s);

        // Discard previous buffer
        b.reset();

        b = inverse_cwt( *chunk );
    }

    BOOST_FOREACH( pSink sink, sinks )
        sink->put(b, s);
}

void PostSink::
        reset()
{
//    BOOST_FOREACH( pSink s, sinks )
//        s->reset( );

    sinks.clear();
}

bool PostSink::
        finished()
{
    bool r = true;

    BOOST_FOREACH( pSink s, sinks )
        r &= s->finished( );

    return r;
}

SamplesIntervalDescriptor PostSink::
        expected_samples()
{
    SamplesIntervalDescriptor x;
    BOOST_FOREACH( pSink s, sinks )
        x |= s->expected_samples();

    return x;
}

void PostSink::
        add_expected_samples( SamplesIntervalDescriptor x )
{
    BOOST_FOREACH( pSink s, sinks )
        s->add_expected_samples( x );
}

} // namespace Signal
