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


pBuffer PostSink::
        read( const Interval& I )
{
    gcSinks();

    pBuffer b;
#ifndef SAWE_NO_MUTEX
    b = readDirect( I );
    //b = readSimple( I );
#else
    b = readActivePassive( I );
#endif

    return b;
}


void PostSink::
        gcSinks()
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
}


pBuffer PostSink::
        readDirect( const Interval& I )
{
    return readDirectSource()->read( I );
}


pBuffer PostSink::
        readSimple( const Interval& I )
{
    DEBUG_POSTSINK TaskTimer tt("%s( %s )", __FUNCTION__, I.toString().c_str());

    pOperation prev = source();
    DEBUG_POSTSINK TaskTimer("Source %s", vartype(*prev).c_str()).suppressTiming();

    if (_filter)
    {
        DEBUG_POSTSINK TaskTimer("Filter %s", vartype(*_filter).c_str()).suppressTiming();
        prev = _filter;
    }

    pBuffer b = prev->read( I );

    BOOST_FOREACH( pOperation c, sinks() ) {
        c->read( I );
    }

    return b;
}


pBuffer PostSink::
        readActivePassive( const Interval& I )
{
    DEBUG_POSTSINK TaskTimer tt("%s( %s )", __FUNCTION__, I.toString().c_str());

    if (_sinks.empty())
        return source()->read(I);

    vector<pOperation> passive_operations;
    vector<pOperation> active_operations;

    {
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_sinks_lock);
#endif

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
    // Also, should that change PostSink implements affected_samples to notify
    // nested PostSinks that they can or can not use the read from a PostSink
    // directly.

    if (1==active_operations.size())
    {
        pOperation c = active_operations[0];
        c->source( prev );
        b = c->read( I );
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


pOperation PostSink::
        readDirectSource()
{
    vector<pOperation> s = sinks();
    if (!s.empty())
        return s.back();

    EXCEPTION_ASSERT( false );

    if (!_filter)
        return _filter;

    return source();
}


pOperation PostSink::
        source()
{
    return Sink::source();
}


void PostSink::
        source(pOperation v)
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_sinks_lock);
#endif
    DeprecatedOperation::source( v );

    update_source();
}


void PostSink::
        update_source()
{
#ifndef SAWE_NO_MUTEX
    // make sure to lock _sinks_lock outside this
    EXCEPTION_ASSERT( false == _sinks_lock.tryLock() );
#endif

    pOperation v = source();

    if (_filter)
    {
        _filter->source( v );
        v = _filter;
    }

    BOOST_FOREACH( pOperation s, _sinks )
    {
        s->source( v );
        v = s;
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
    pOperation s = source();
    if (s)
    {
        // prevent endless recursion calls with 'noop'
        static bool noop = false;
        if (noop)
            return;

    #ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_sinks_lock);
    #endif

        noop = true;
        s->Operation::invalidate_samples( I );
        noop = false;
    }
    else
    {
        if (_filter) _filter->invalidate_samples( I );

        BOOST_FOREACH( pOperation o, sinks() )
            o->invalidate_samples( I );
    }
}


std::string PostSink::
        toString()
{
    std::stringstream ss;
    ss << name() << ", " << sinks().size() << " sink" << (sinks().size()!=1?"s":"");

    if (_filter)
    {
        ss << std::endl << "Filter: " << _filter->toStringSkipSource();
    }
    else
        ss << ". No filter";

    unsigned i = 0;
    BOOST_FOREACH( pOperation o, sinks() )
    {
        ss << std::endl << "Sink " << i << ": " << o->toStringSkipSource() << "[/" << i << "]";
        i++;
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
    EXCEPTION_ASSERT( v.size() <= 1 );

#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_sinks_lock);
#endif

    // validate that they are all sinks
    BOOST_FOREACH( pOperation o, v )
    {
        Sink* s = dynamic_cast<Sink*>(o.get());
        EXCEPTION_ASSERT( s );
    }

    _sinks = v;

    update_source();

    // The new sinks should not be invalidated, neither should the parents be
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

#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_sinks_lock);
#endif

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

    // Change filter
    _filter = f;

    update_source();

#ifndef SAWE_NO_MUTEX
    l.unlock();
#endif

    invalidate_samples( I );
}

} // namespace Signal
