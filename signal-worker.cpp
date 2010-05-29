#include "signal-worker.h"
#include "signal-samplesintervaldescriptor.h"
#include "signal-filteroperation.h"
#include <QTime>
#include <QMutexLocker>
#include <boost/foreach.hpp>
#include <CudaException.h>

namespace Signal {

Worker::
        Worker(Signal::pSource s)
:   _source(s),
    _samples_per_chunk( 1<<12 ),
    _max_samples_per_chunk( 1<<16 )
{
    // Could create an first estimate of _samples_per_chunk based on available memory
    // unsigned mem = CudaProperties::getCudaDeviceProp( CudaProperties::getCudaCurrentDevice() ).totalGlobalMem;
    // but 1<< 12 works well on most GPUs
}


///// OPERATIONS


bool Worker::
        workOne( unsigned middle_chunk )
{
    QTime t;
    t.start();

    // todo_list &= SamplesIntervalDescriptor(0, _source->number_of_samples());

    if (todo_list().isEmpty())
        return false;

    SamplesIntervalDescriptor::Interval interval;
    interval = todo_list().getInterval( _samples_per_chunk, middle_chunk );
    if (interval.first == interval.last) {
        throw std::invalid_argument(std::string(__FUNCTION__) + " todo_list.getInterval returned interval.first == interval.last" );
    }

    pBuffer b;

    try {
        b = _source->read( interval.first, interval.last-interval.first );
        QMutexLocker l(&_todo_lock);
        _todo_list -= b->getInterval();
    } catch (const CudaException& e ) {
        if (cudaErrorMemoryAllocation == e.getCudaError() && 1<_samples_per_chunk) {
            _samples_per_chunk >>=1;
            _max_samples_per_chunk = _samples_per_chunk;
        } else {
            throw;
        }
    }

    callCallbacks( b );

    int milliseconds = t.elapsed();
    if (0==milliseconds) milliseconds=1;

    if (0) {
        if (1000.f/milliseconds < _requested_fps && _samples_per_chunk>1024)
        {
            if (1<_samples_per_chunk) {
                _samples_per_chunk>>=1;
            }
        }
        else if (1000.f/milliseconds > 2.5f*_requested_fps)
        {
            _samples_per_chunk<<=1;
            if (_samples_per_chunk>_max_samples_per_chunk)
                _samples_per_chunk=_max_samples_per_chunk;
        }
    }

    return true;
}


///// PROPERTIES

void Worker::
        todo_list( Signal::SamplesIntervalDescriptor v )
{
    {
        QMutexLocker l(&_todo_lock);
        _todo_list = v;
    }
    _todo_condition.wakeAll();
}

Signal::SamplesIntervalDescriptor Worker::
        todo_list()
{
    QMutexLocker l(&_todo_lock);
    return _todo_list;
}

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
        run()
{
    while (true)
    {
        while(!todo_list().isEmpty())
        {
            workOne( center );
        }
        _todo_condition.wait( &_todo_lock );
    }
}

void Worker::
        addCallback( pSink c )
{
    QMutexLocker l( &_callbacks_lock );
    _callbacks.push_back( c );
}

void Worker::
        removeCallback( pSink c )
{
    QMutexLocker l( &_callbacks_lock );
    _callbacks.remove( c );
}

void Worker::
        callCallbacks( pBuffer b )
{
    std::list<pSink> worklist;
    {
        QMutexLocker l( &_callbacks_lock );
        worklist = _callbacks;
    }

    BOOST_FOREACH( pSink c, worklist ) {
        c->put( b, _source );
    }

    b->waveform_data->getCpuMemory();
    b->waveform_data->freeUnused();

    FilterOperation* f = dynamic_cast<FilterOperation*>(_source.get());
    if (f)
        f->release_previous_chunk();
}

} // namespace Signal
