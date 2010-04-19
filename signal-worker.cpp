#include "signal-worker.h"

namespace Signal {

Worker::
Worker(Signal::pSource s)
:   _source(s)
{
}


///// OPERATIONS


void Worker::
workOne()
{
    float a=0, b=0;
    QTime t();
    t.start();

    todo_list.get( _samples_per_chunk, a, b );
    pBuffer b = _source->read( a, b-a );

    callCallbacks(b);

    int milliseconds = t.elapsed();
    if (0==milliseconds) milliseconds=1;

    if (1000.f/milliseconds < _requested_fps && _samples_per_chunk>1024)
        _samples_per_chunk>>=1;
    else if (1000.f/milliseconds > 2.5f*_requested_fps)
        _samples_per_chunk<<=1;
}


///// PROPERTIES


Signal::pSource Worker::
source() const
{
    return _source;
}

void Worker::
source(Signal::pSource value)
{
    _source = value;
    // todo_list.clear();
}
unsigned Worker::
samples_per_chunk() const
{
    return _samples_per_chunk;
}

unsigned Worker::
requested_fps() const
{
    return _requested_fps;
}


///// PRIVATE


void Worker::
requested_fps(unsigned value)
{
    _requested_fps = value>0?value:1;
}

void Worker::
addCallback( WorkerCallback* c )
{
    QMutexLocker( &_mutex );
    _callbacks.push_back( c );
}

void Worker::
removeCallback( WorkerCallback* c )
{
    QMutexLocker( &_mutex );
    _callbacks.remove( c );
}

void Worker::
callCallbacks( pBuffer b )
{
    QMutexLocker( &_mutex );
    BOOST_FOR_EACH( WorkerCallback* c, _callbacks ) {
        c->put( b, _source );
    }
}

} // namespace Signal
