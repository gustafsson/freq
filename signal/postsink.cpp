#include "postsink.h"
#include "buffersource.h"
#include "tfr/filter.h"
#include <boost/foreach.hpp>
#include <demangle.h>
#include <typeinfo>

using namespace std;

namespace Signal {

Signal::pBuffer PostSink::
        read( const Signal::Interval& I )
{
    TaskTimer tt("%s in %s", __FUNCTION__, demangle(typeid(*this).name()).c_str());

    vector<pOperation> passive_operations;
    vector<pOperation> active_operations;

    {
        QMutexLocker l(&_sinks_lock);

        BOOST_FOREACH( pOperation c, _sinks )
        {
            Operation* s = (Operation*)c.get();
            if (s->affected_samples() & I )
                active_operations.push_back(c);
            else
                passive_operations.push_back(c);
        }
    }

    pOperation prev = source();
    BOOST_FOREACH( pOperation c, passive_operations) {
        c->source(prev);
        prev = c;
    }

    if (1==active_operations.size())
    {
        pOperation c = active_operations.front();
        c->source(prev);
        prev = c;
        active_operations.clear();
    }

    pBuffer b = prev->read( I );

    if (!active_operations.empty())
    {
        prev.reset( new BufferSource( b ));
        BOOST_FOREACH( pOperation c, active_operations) {
            c->source(prev);
            prev = c;
        }
        prev->read( I );
    }

    return b;
}

/* TODO remove
void PostSink::
        operator()( Chunk& chunk )
{
    Signal::Intervals expected = expected_samples();
    if ( !(expected & chunk.getInterval()) )
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
}*/


/* TODO remove
void PostSink::
        reset()
{
    BOOST_FOREACH( pSink s, sinks() )
        s->reset( );

    Tfr::ChunkSink::reset();
}*/


/* TODO remove
bool PostSink::
        isFinished()
{
    bool r = true;

    BOOST_FOREACH( pSink s, sinks() )
        r &= s->isFinished( );

    return r;
}
*/


Intervals PostSink::
        affected_samples()
{
    Intervals I = Operation::affected_samples();

    if (_filter)
        I |= _filter->affected_samples();

    BOOST_FOREACH( pOperation s, sinks() )
    {
        Operation* o = dynamic_cast<Operation*>(s.get());
        if (o)
            I |= o->affected_samples();
    }

    return I;
}


Intervals PostSink::
        invalid_samples()
{
    Intervals I = Operation::invalid_samples();

    BOOST_FOREACH( pOperation s, sinks() )
    {
        Operation* o = dynamic_cast<Operation*>(s.get());
        if (o)
            I |= o->invalid_samples();
    }

    return I;
}


void PostSink::
        invalidate_samples( const Intervals& I )
{
    BOOST_FOREACH( pOperation s, sinks() )
    {
        Operation* o = dynamic_cast<Operation*>(s.get());
        if (o)
            o->invalidate_samples( I );
    }

    Operation::invalidate_samples( I );
}


std::vector<pOperation> PostSink::
        sinks()
{
    QMutexLocker l(&_sinks_lock);
    return _sinks;
}


void PostSink::
        sinks(std::vector<pOperation> v)
{
    QMutexLocker l(&_sinks_lock);
    _sinks = v;
}


pOperation PostSink::
        filter()
{
    return _filter;
}


void PostSink::
        filter(pOperation f)
{
    Intervals I;

    I |= f->affected_samples();
    I |= _filter->affected_samples();

    Tfr::Filter* filt = dynamic_cast<Tfr::Filter*>(f.get());
    if (filt)
        I -= filt->zeroed_samples();

    invalidate_samples( I );

    // Change filter
    _filter = f;
}

} // namespace Signal
