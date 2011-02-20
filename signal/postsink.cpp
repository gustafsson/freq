#include "postsink.h"

#include "buffersource.h"

// gpumisc
#include <Statistics.h>
#include <TaskTimer.h>
#include <demangle.h>

// boost
#include <boost/foreach.hpp>

// std
#include <typeinfo>


#define DEBUG_POSTSINK if(0)
//#define DEBUG_POSTSINK


using namespace std;

namespace Signal {

PostSink::
        PostSink()
            :
            isUnderfedIfInvalid(false)
{
}


Signal::pBuffer PostSink::
        read( const Signal::Interval& I )
{
    DEBUG_POSTSINK TaskTimer tt("PostSink( %s )", I.toString().c_str());

    if (_sinks.empty())
        return source()->read(I);

    vector<pOperation> passive_operations;
    vector<pOperation> active_operations;

    {
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_sinks_lock);
#endif

        for(std::vector<pOperation>::iterator i = _sinks.begin(); i!=_sinks.end(); )
        {
            Sink* s = dynamic_cast<Sink*>(i->get());

            if (s && s->deleteMe())
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

    pBuffer b;
    // Since PostSink is a sink, it doesn't need to return anything.
    // But since the buffer 'b' will be computed anyway when calling 'read'
    // PostSink may just as well return it, at least for debugging purposes.

    if (1==active_operations.size())
    {
        pOperation c = active_operations[0];
        c->source( prev );
        b = c->read( I );
        prev = c;
        c->source( source() );
        active_operations.clear();
    } else
        b = prev->read( I );

    // prev.reset( new BufferSource( b ));
    BOOST_FOREACH( pOperation c, active_operations) {
        c->source( _filter ? _filter : source() );
        c->read( I );
    }

    BOOST_FOREACH( pOperation c, _sinks )
    {
        c->source( source() );
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

    foreach( pSink sink, sinks() )
        sink->put(b, s);
}*/


/* TODO remove
void PostSink::
        reset()
{
    foreach( pSink s, sinks() )
        s->reset( );

    Tfr::ChunkSink::reset();
}*/


/* TODO remove
bool PostSink::
        isFinished()
{
    bool r = true;

    foreach( pSink s, sinks() )
        r &= s->isFinished( );

    return r;
}
*/


void PostSink::
        set_channel(unsigned c)
{
    Operation::set_channel( c );

    if (_filter)
    {
        _filter->source( pOperation() );
        _filter->set_channel( c );
        _filter->source( source() );
    }

    BOOST_FOREACH( pOperation s, sinks() )
    {
        s->source( pOperation() );
        s->set_channel( c );
        s->source( source() );
    }
}


void PostSink::
        source(pOperation v)
{
    Operation::source( v );

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

    if (sinks().empty())
        return I;

    if (_filter)
        I |= _filter->affected_samples();

    BOOST_FOREACH( pOperation s, sinks() )
    {
        I |= s->affected_samples();
    }

    return I;
}


Intervals PostSink::
        invalid_samples()
{
    Intervals I;

    BOOST_FOREACH( pOperation o, sinks() )
    {
        // Sinks doesn't fetch invalid sampels recursively
        Sink* s = dynamic_cast<Sink*>(o.get());
        I |= s->invalid_samples();
    }

    return I;
}


bool PostSink::
        isUnderfed()
{
    bool r = false;

    BOOST_FOREACH( pOperation o, sinks() )
    {
        Sink* s = dynamic_cast<Sink*>(o.get());
        r = r || s->isUnderfed();
    }

    if (isUnderfedIfInvalid)
        r = r || invalid_samples();

    return r;
}


void PostSink::
        invalidate_samples( const Intervals& I )
{
    BOOST_FOREACH( pOperation o, sinks() )
    {
        Sink* s = dynamic_cast<Sink*>(o.get());
        s->invalidate_samples( I );
    }

    Operation::invalidate_samples( I );
}


std::string PostSink::
        toString()
{
    std::stringstream ss;
    ss << name() << ", " << sinks().size() << " sink" << (sinks().size()!=1?"s":"");

    if (_filter)
    {
        _filter->source( pOperation() );
        ss << std::endl << "Filter: " << _filter->toString();
        _filter->source( source() );
    }
    else
        ss << ". No filter";

    unsigned i = 0;
    BOOST_FOREACH( pOperation o, sinks() )
    {
        o->source( pOperation() );
        ss << std::endl << "Sink " << i << ": " << o->toString() << "[/" << i << "]";
        i++;
        o->source( source() );
    }

    if (source())
        ss << std::endl << source()->toString();

    return ss.str();
}


std::vector<pOperation> PostSink::
        sinks()
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_sinks_lock);
#endif
    return _sinks;
}


void PostSink::
        sinks(std::vector<pOperation> v)
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_sinks_lock);
#endif

    BOOST_FOREACH( pOperation o, v )
    {
        Sink* s = dynamic_cast<Sink*>(o.get());
        BOOST_ASSERT( s );
        s->source( source() );
    }

    _sinks = v;

    invalidate_samples( Signal::Intervals::Intervals_ALL );
}


pOperation PostSink::
        filter()
{
    return _filter;
}


void PostSink::
        filter(pOperation f)
{
    if (f == _filter)
        return;

    Intervals I;

    if (f)          f->source(source());
    if (_filter)    _filter->source(source());

    if (f)          I |= f->affected_samples();
    if (_filter)    I |= _filter->affected_samples();

    if (f && _filter)
        I -= f->zeroed_samples_recursive() & _filter->zeroed_samples_recursive();
    else if(f)
        I -= f->zeroed_samples_recursive();
    else if(_filter)
        I -= _filter->zeroed_samples_recursive();

    invalidate_samples( I );

    // Change filter
    _filter = f;
}

} // namespace Signal
