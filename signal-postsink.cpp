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
        Tfr::pChunk chunk;

        // If buffer comes directly from a Signal::FilterOperation
        Signal::FilterOperation* filterOp = dynamic_cast<Signal::FilterOperation*>(s.get());
        if (filterOp) {
            // use the Cwt chunk still stored in FilterOperation
            chunk = filterOp->previous_chunk();
            tt.info("Stealing filterOp chunk. Got %p", chunk.get());

            if (0 == chunk) {
                // try again
                filterOp->read( b->sample_offset, b->number_of_samples() );
                chunk = filterOp->previous_chunk();

                tt.info("Failed, tried again. Got %p", chunk.get());
            }
        }

        if (0 == chunk) {
            // otherwise compute the Cwt of this block
            chunk = Tfr::CwtSingleton::operate( b );
            tt.info("Computing raw chunk. Got %p", chunk.get());

            // Should either read extra data off from source or just compute a smaller chunk...
            chunk->transform_data.reset();
            chunk->n_valid_samples = chunk->transform_data->getNumberOfElements().width;
            chunk->first_valid_sample = 0;
        }

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
