#include "signal-worker.h"
#include "signal-samplesintervaldescriptor.h"
#include "signal-filteroperation.h"
#include <QTime>
#include <QMutexLocker>
#include <boost/foreach.hpp>
#include <CudaException.h>
#include "tfr-cwt.h"

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
    todo_list( SamplesIntervalDescriptor() );
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

    SamplesIntervalDescriptor::Interval interval;
    interval = todo_list().getInterval( _samples_per_chunk, center_sample );

    pBuffer b;

    try
    {
        {
            stringstream ss;
            TIME_WORKER TaskTimer tt("Reading source %s", ((std::stringstream&)(ss<<interval)).str().c_str() );

            b = _source->readFixedLength( interval.first, interval.last-interval.first );
            work_time += b->length();
            QMutexLocker l(&_todo_lock);
            _todo_list -= b->getInterval();
        }

        {   stringstream ss;
            TIME_WORKER TaskTimer tt("Calling callbacks %s", ((std::stringstream&)(ss<<b->getInterval())).str().c_str() );

            callCallbacks( b );
        }

        CudaException_CHECK_ERROR();
    } catch (const CudaException& e ) {
        if (cudaErrorMemoryAllocation == e.getCudaError() && 1<_samples_per_chunk) {
            _samples_per_chunk = Tfr::CwtSingleton::instance()->prev_good_size(
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
        unsigned minSize = Tfr::CwtSingleton::instance()->next_good_size( 1, _source->sample_rate());
        float current_fps = 1000.f/milliseconds;
        if (current_fps < _requested_fps && _samples_per_chunk >= minSize)
        {
            _samples_per_chunk = Tfr::CwtSingleton::instance()->prev_good_size(
                    _samples_per_chunk, _source->sample_rate());
            TIME_WORKER TaskTimer("Low framerate (%.1f fps). Decreased samples per chunk to %u", 1000.f/milliseconds, _samples_per_chunk).suppressTiming();
        }
        else if (current_fps > 2.5f*_requested_fps && _samples_per_chunk <= b->number_of_samples() && _samples_per_chunk < _max_samples_per_chunk)
        {
            _samples_per_chunk = Tfr::CwtSingleton::instance()->next_good_size(
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
        todo_list( Signal::SamplesIntervalDescriptor v )
{
    {
        QMutexLocker l(&_todo_lock);
        _todo_list = v;
    }
    if (!v.isEmpty())
        _todo_condition.wakeAll();
}

Signal::SamplesIntervalDescriptor Worker::
        todo_list()
{
    QMutexLocker l(&_todo_lock);
    Signal::SamplesIntervalDescriptor c = _todo_list;
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


///// PRIVATE


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
    list<pSink> worklist;
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
