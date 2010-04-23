#include "signal-worker.h"
#include "signal-samplesintervaldescriptor.h"
#include <QTime>
#include <QMutexLocker>
#include <boost/foreach.hpp>

namespace Signal {

Worker::
        Worker(Signal::pSource s)
:   _source(s),
    _samples_per_chunk( 1<< 12 )
{
    // Could create an first estimate of _samples_per_chunk based on available memory
    // unsigned mem = CudaProperties::getCudaDeviceProp( CudaProperties::getCudaCurrentDevice() ).totalGlobalMem;
    // but 1<< 12 works well on most GPUs
}


///// OPERATIONS


void Worker::
        workOne()
{
    QTime t;
    t.start();

    SamplesIntervalDescriptor::Interval interval;
    interval = todo_list.popInterval( _samples_per_chunk, 0 );
    if (interval.first == interval.last)
        return;

    pBuffer b = _source->read( interval.first, interval.last-interval.first );

    callCallbacks( b );

    int milliseconds = t.elapsed();
    if (0==milliseconds) milliseconds=1;

    if (1000.f/milliseconds < _requested_fps && _samples_per_chunk>1024)
    {
        if (1<_samples_per_chunk) {
            _samples_per_chunk>>=1;
        }
    }
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
        addCallback( Sink* c )
{
    QMutexLocker l( &_lock );
    _callbacks.push_back( c );
}

void Worker::
        removeCallback( Sink* c )
{
    QMutexLocker l( &_lock );
    _callbacks.remove( c );
}

void Worker::
        callCallbacks( pBuffer b )
{
    QMutexLocker l( &_lock );

    BOOST_FOREACH( Sink* c, _callbacks ) {
        c->put( b, _source.get() );
    }
}

} // namespace Signal
