#include "worker.h"
#include "intervals.h"
#include "cwtfilter.h"
#include "buffersource.h"

#include "tfr/cwt.h" // hack to make chunk sizes adapt to computer speed and memory

#include <QTime>
#include <QMutexLocker>
#include <boost/foreach.hpp>
#include <CudaException.h>

//#define TIME_WORKER
#define TIME_WORKER if(0)

#define TESTING_PERFORMANCE false

using namespace std;
using namespace boost::posix_time;

namespace Signal {

Worker::
        Worker(Signal::pSource s)
:   work_chunks(0),
    work_time(0),
    _source(s),
    _samples_per_chunk( 1<<12 ),
    _max_samples_per_chunk( 1<<16 ),
    _requested_fps( 20 ),
    _caught_exception( "" ),
    _caught_invalid_argument("")
{
    // Could create an first estimate of _samples_per_chunk based on available memory
    // unsigned mem = CudaProperties::getCudaDeviceProp( CudaProperties::getCudaCurrentDevice() ).totalGlobalMem;
    // but 1<< 12 works well on most GPUs
}

Worker::
        ~Worker()
{
    this->quit();
    todo_list( Intervals() );
}

///// OPERATIONS


bool Worker::
        workOne()
{
    if (todo_list().isEmpty())
        return false;

    if (TESTING_PERFORMANCE) _samples_per_chunk = 19520;
    work_chunks++;

    unsigned center_sample = source()->sample_rate() * center;
    stringstream ss;
    TIME_WORKER TaskTimer tt("Working %s from %u", ((std::stringstream&)(ss<<todo_list())).str().c_str(), center_sample );

    ptime startTime = microsec_clock::local_time();

    Interval interval;
    interval = todo_list().getInterval( _samples_per_chunk, center_sample );

    pBuffer b;

    try
    {
        {   stringstream ss;
            TIME_WORKER TaskTimer tt("Reading source and calling callbacks %s", ((std::stringstream&)(ss<<interval)).str().c_str() );

            pBuffer b = callCallbacks( interval );
            work_time += b->length();
        }

        CudaException_CHECK_ERROR();
    } catch (const CudaException& e ) {
        if (cudaErrorMemoryAllocation == e.getCudaError() && 1<_samples_per_chunk) {
            _samples_per_chunk = Tfr::Cwt::Singleton().prev_good_size(
                    _samples_per_chunk, _source->sample_rate());
            _max_samples_per_chunk = _samples_per_chunk;
            TaskTimer("Worker caught cudaErrorMemoryAllocation. Setting max samples per chunk to %u\n%s", _samples_per_chunk, e.what()).suppressTiming();
        } else {
            throw;
        }
    }

    time_duration diff = microsec_clock::local_time() - startTime;

    unsigned milliseconds = diff.total_milliseconds();
    if (0==milliseconds) milliseconds=1;

    if (!TESTING_PERFORMANCE) {
        unsigned minSize = Tfr::Cwt::Singleton().next_good_size( 1, _source->sample_rate());
        float current_fps = 1000.f/milliseconds;
        if (current_fps < _requested_fps && _samples_per_chunk >= minSize)
        {
            _samples_per_chunk = Tfr::Cwt::Singleton().prev_good_size(
                    _samples_per_chunk, _source->sample_rate());
            TIME_WORKER TaskTimer("Low framerate (%.1f fps). Decreased samples per chunk to %u", 1000.f/milliseconds, _samples_per_chunk).suppressTiming();
        }
        else if (current_fps > 2.5f*_requested_fps && _samples_per_chunk <= b->number_of_samples() && _samples_per_chunk < _max_samples_per_chunk)
        {
            _samples_per_chunk = Tfr::Cwt::Singleton().next_good_size(
                    _samples_per_chunk, _source->sample_rate());
            if (_samples_per_chunk>_max_samples_per_chunk)
                _samples_per_chunk=_max_samples_per_chunk;
            else
                TIME_WORKER TaskTimer("High framerate (%.1f fps). Increased samples per chunk to %u", 1000.f/milliseconds, _samples_per_chunk).suppressTiming();
        }

        _requested_fps = 99*_requested_fps/100;
        //if (1>_requested_fps)_requested_fps=1;
    }

    return true;
}


///// PROPERTIES

void Worker::
        todo_list( Signal::Intervals v )
{
    {
        QMutexLocker l(&_todo_lock);
        _todo_list = v;
    }
    if (!v.isEmpty())
        _todo_condition.wakeAll();
}

Signal::Intervals Worker::
        todo_list()
{
    QMutexLocker l(&_todo_lock);
    Signal::Intervals c = _todo_list;
    return c;
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

void Worker::
		samples_per_chunk_hint(unsigned value)
{
	_samples_per_chunk = value;
}

unsigned Worker::
        requested_fps() const
{
    return _requested_fps;
}


void Worker::
        requested_fps(unsigned value)
{
    if (0==value) value=1;

    if (value>_requested_fps) {
        _requested_fps = value;
    }
}


void Worker::
		checkForErrors()
{
	if (_caught_invalid_argument.what() && 0 != *_caught_invalid_argument.what()) {
		invalid_argument x = _caught_invalid_argument;
		_caught_invalid_argument = invalid_argument("");
		throw x;
	}
	
	if (_caught_exception.what() && 0 != *_caught_exception.what()) {
                runtime_error x = _caught_exception;
                _caught_exception = runtime_error("");
		throw x;
	}
}


std::vector<pSource> Worker::
        callbacks()
{
    QMutexLocker l(&_callbacks_lock);
    std::vector<pSource> c = _callbacks;
    return c;
}


///// PRIVATE


void Worker::
        run()
{
	while (true)
	{
		try {
			while(!todo_list().isEmpty())
			{
				workOne();
				msleep(1);
			}
		} catch ( const std::invalid_argument& x ) {
			if (0 == *_caught_invalid_argument.what())
				_caught_invalid_argument = x;
		} catch ( const std::exception& x ) {
			if (0 == _caught_exception.what())
                                _caught_exception = runtime_error(x.what());
		} catch ( ... ) {
			if (0 == _caught_exception.what())
				_caught_exception = std::runtime_error("Unknown exception");
		}

		try {
			{ TIME_WORKER TaskTimer tt("Worker is waiting for more work to do");
            QMutexLocker l(&_todo_lock);
			_todo_condition.wait( &_todo_lock );}
		} catch ( const std::exception& x ) {
                        _caught_exception = runtime_error(x.what());
			return;
		}
	}
}

void Worker::
        addCallback( pSink c )
{
    BOOST_ASSERT(dynamic_cast<Sink*>(c.get()));

    QMutexLocker l( &_callbacks_lock );
    _callbacks.push_back( c );
}

void Worker::
        removeCallback( pSink c )
{
    BOOST_ASSERT(dynamic_cast<Sink*>(c.get()));

    QMutexLocker l( &_callbacks_lock );
    _callbacks.remove( c );
}

pBuffer Worker::
        callCallbacks( Interval i )
{
    vector<pSource> worklist;
    {
        QMutexLocker l( &_callbacks_lock );
        worklist = _callbacks;
    }
    vector<pSource> passive_operations;
    vector<pSource> active_operations;

    BOOST_FOREACH( pSource c, worklist ) {
        Sink* s = (Sink*)c.get();
        if (s->nonpassive_operation() & I )
            active_operations.push_back(c);
        else
            passive_operations.push_back(c);
    }

    pSource prev = source();
    BOOST_FOREACH( pSource c, passive_operations) {
        Sink* s = (Sink*)c.get();
        s->source(prev);
        prev = c;
    }

    if (1==active_operations.size())
    {
        Sink* s = (Sink*)active_operations.front();
        s->source(prev);
        prev = c;
        active_operations.clear();
    }

    pBuffer b = prev->read( I );

    QMutexLocker l(&_todo_lock);
    _todo_list -= b->getInterval();

    if (!active_operations.empty())
    {
        prev.reset( new BufferSource( b ));
        BOOST_FOREACH( pSource c, active_operations) {
            Sink* s = (Sink*)c.get();
            s->source(prev);
            prev = c;
        }
        prev->read( I );
    }

    return b;
}

} // namespace Signal
