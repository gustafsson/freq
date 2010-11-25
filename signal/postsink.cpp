#include "postsink.h"
#include "buffersource.h"
#include <boost/foreach.hpp>
#include <demangle.h>
#include <typeinfo>

#include <Statistics.h>
#include <TaskTimer.h>
#include <demangle.h>

#define DEBUG_POSTSINK if(0)
//#define DEBUG_POSTSINK

using namespace std;

namespace Signal {

Signal::pBuffer PostSink::
        read( const Signal::Interval& I )
{
    DEBUG_POSTSINK TaskTimer tt("PostSink( %s )", I.toString().c_str());

    vector<pOperation> passive_operations;
    vector<pOperation> active_operations;

    {
        QMutexLocker l(&_sinks_lock);

        for(std::vector<pOperation>::iterator i = _sinks.begin(); i!=_sinks.end(); )
        {
            Sink* s = dynamic_cast<Sink*>(i->get());

            if (s && s->isFinished())
            {
                TaskTimer tt("Removing %s from postsink", demangle( typeid(*s) ).c_str());
                i = _sinks.erase( i );
            }
            else
                i++;
        }

        DEBUG_POSTSINK TaskTimer tt("Adding %u operations", _sinks.size());

        BOOST_FOREACH( pOperation c, _sinks )
        {
            if (c->affected_samples() & I )
            {
                DEBUG_POSTSINK TaskTimer("Active %s", vartype(*c).c_str()).suppressTiming();
                active_operations.push_back(c);
            }
            else
            {
                DEBUG_POSTSINK TaskTimer("Passive %s", vartype(*c).c_str()).suppressTiming();
                passive_operations.push_back(c);
            }
        }
    }

    pOperation prev = source();
    DEBUG_POSTSINK TaskTimer("Source %s", vartype(*prev).c_str()).suppressTiming();

    if (_filter)
    {
        DEBUG_POSTSINK TaskTimer("Filter %s", vartype(*_filter).c_str()).suppressTiming();
        _filter->source(prev);
        prev = _filter;
    }

    BOOST_FOREACH( pOperation c, passive_operations) {
        c->source(prev);
        prev = c;
    }

    pBuffer b = prev->read( I );

    // prev.reset( new BufferSource( b ));
    BOOST_FOREACH( pOperation c, active_operations) {
        c->source( prev );
        c->read( I );
    }

    BOOST_FOREACH( pOperation c, passive_operations )
        c->source(source());

    BOOST_FOREACH( pOperation c, active_operations )
        c->source(source());


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


void PostSink::
        source(pOperation v)
{
    Operation::source(v);

    if (_filter)
        _filter->source( v );

    BOOST_FOREACH( pOperation s, sinks() )
    {
        s->source( v );
    }
}


Intervals PostSink::
        affected_samples()
{
    Intervals I;

    if (_filter)
        I |= _filter->affected_samples();

    BOOST_FOREACH( pOperation s, sinks() )
    {
        I |= s->affected_samples();
    }

    return I;
}


Intervals PostSink::
        fetch_invalid_samples()
{
    Intervals I;

    BOOST_FOREACH( pOperation s, sinks() )
    {
        // Sinks doesn't fetch invalid sampels recursively
        I |= s->fetch_invalid_samples();
    }

    return I;
}


void PostSink::
        invalidate_samples( const Intervals& I )
{
    BOOST_FOREACH( pOperation o, sinks() )
    {
        Sink* s = dynamic_cast<Sink*>(o.get());

        if (s)
            s->invalidate_samples( I );
    }
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

    f->source(source());

    if (f)          I |= f->affected_samples();
    if (_filter)    I |= _filter->affected_samples();

    if (f && _filter)
        I -= f->zeroed_samples() & _filter->zeroed_samples();
    else if(f)
        I -= f->zeroed_samples();
    else if(_filter)
        I -= _filter->zeroed_samples();

    invalidate_samples( I );

    // Change filter
    _filter = f;
}

} // namespace Signal
