#include "worker.h"
#include "intervals.h"
#include "postsink.h"
#include "operationcache.h"

#include "tfr/cwt.h" // hack to make chunk sizes adapt to computer speed and memory

#include <QTime>
#include <QMutexLocker>
#include <CudaException.h>
#include <demangle.h>

//#define TIME_WORKER
#define TIME_WORKER if(0)

//#define WORKER_INFO
#define WORKER_INFO if(0)

#define TESTING_PERFORMANCE false

using namespace std;
using namespace boost::posix_time;

namespace Signal {

Worker::
        Worker(Signal::pOperation s)
:   work_chunks(0),
    work_time(0),
    _last_work_one(boost::date_time::not_a_date_time),
    _source(s),
    _cache( new OperationCacheLayer(pOperation()) ),
    _samples_per_chunk( 1 ),
    _max_samples_per_chunk( (unsigned)-1 ),
    _requested_fps( 20 ),
    _min_fps( 0.5 ),
    _caught_exception( "" ),
    _caught_invalid_argument("")
{
    source( s );
    // Could create an first estimate of _samples_per_chunk based on available memory
    // unsigned mem = CudaProperties::getCudaDeviceProp( CudaProperties::getCudaCurrentDevice() ).totalGlobalMem;
    // but 1<< 12 works well on most GPUs
}

Worker::
        ~Worker()
{
    TaskInfo tt(__FUNCTION__);

    this->quit();
    todo_list( Intervals() );
}

///// OPERATIONS


bool Worker::
        workOne()
{
    if (todo_list().empty())
        return false;

    // todo_list().print(__FUNCTION__);

    if (TESTING_PERFORMANCE) _samples_per_chunk = _max_samples_per_chunk;
    work_chunks++;

    unsigned center_sample = source()->sample_rate() * center;

    Interval interval = todo_list().getInterval( _samples_per_chunk, center_sample );
    _cheat_work |= interval;

    boost::scoped_ptr<TaskTimer> tt;

    TIME_WORKER {
        tt.reset( new TaskTimer(
                "Working %u samples from %s, center=%u. Reading %s",
                _samples_per_chunk,
                todo_list().toString().c_str(),
                center_sample,
                interval.toString().c_str()));
    }


    pBuffer b;

    try
    {
        CudaException_CHECK_ERROR();

        b = callCallbacks( interval );

        //float r = 2*Tfr::Cwt::Singleton().wavelet_time_support_samples( b->sample_rate )/b->sample_rate;
        work_time += b->length();
        //work_time -= r;

        WORKER_INFO {
            tt->info("Worker got %s, [%g, %g) s. %g x realtime",
                b->getInterval().toString().c_str(),
                b->start(), b->start()+b->length(),
                //(b->length()-r)/tt->elapsedTime());
                b->length()/tt->elapsedTime());
        }

        CudaException_CHECK_ERROR();
    } catch (const CudaException& e ) {
        if (cudaErrorMemoryAllocation == e.getCudaError() && 1<_samples_per_chunk) {
            cudaGetLastError(); // consume error
            TaskInfo("_samples_per_chunk was %u", _samples_per_chunk);
            TaskInfo("_max_samples_per_chunk was %u", _max_samples_per_chunk);
            TaskInfo("scales_per_octave was %g", Tfr::Cwt::Singleton().scales_per_octave() );

            while (128 < _samples_per_chunk &&
                   _samples_per_chunk <= Tfr::Cwt::Singleton().prev_good_size(
                    _samples_per_chunk, _source->sample_rate()))
            {
                Tfr::Cwt::Singleton().scales_per_octave( Tfr::Cwt::Singleton().scales_per_octave()*0.99f );
            }

            _samples_per_chunk = Tfr::Cwt::Singleton().prev_good_size(
                    _samples_per_chunk, _source->sample_rate());
            _max_samples_per_chunk = _samples_per_chunk;

            TaskInfo("scales_per_octave is now %g", Tfr::Cwt::Singleton().scales_per_octave() );
            TaskInfo("_samples_per_chunk now is %u", _samples_per_chunk);
            TaskInfo("_max_samples_per_chunk now is %u", _max_samples_per_chunk);
            TaskInfo("Worker caught cudaErrorMemoryAllocation. Setting max samples per chunk to %u\n%s", _samples_per_chunk, e.what());

            size_t free=0, total=0;
            cudaMemGetInfo(&free, &total);
            TaskInfo("Cuda memory available %g MB (of which %g MB is free to use)",
                     total/1024.f/1024, free/1024.f/1024);

        } else {
            TaskInfo("Worker caught CudaException:\n%s", e.what());
            throw;
        }
    } catch (const exception& e) {
        TaskInfo("Worker caught exception type %s:\n%s",
                  vartype(e).c_str(), e.what());
        throw;
    } catch (...) {
        TaskInfo("Worker caught unknown exception.");
        throw;
    }

    ptime now = microsec_clock::local_time();

    if (b && !_last_work_one.is_not_a_date_time()) if (!TESTING_PERFORMANCE) {
        time_duration diff = now - _last_work_one;
        float current_fps = 1000000.0/diff.total_microseconds();

        if (current_fps < _requested_fps &&
            _samples_per_chunk >= _min_samples_per_chunk)
        {
            _samples_per_chunk = Tfr::Cwt::Singleton().prev_good_size(
                    _samples_per_chunk, _source->sample_rate());
            WORKER_INFO TaskInfo(
                    "Low framerate (%.1f fps). Decreased samples per chunk to %u",
                    current_fps, _samples_per_chunk);
        }
        else if (current_fps > 2.5f*_requested_fps)
        {
            _samples_per_chunk = Tfr::Cwt::Singleton().next_good_size(
                    _samples_per_chunk, _source->sample_rate());
            if (_samples_per_chunk>_max_samples_per_chunk)
                _samples_per_chunk=_max_samples_per_chunk;
            else
                WORKER_INFO TaskInfo(
                        "High framerate (%.1f fps). Increased samples per chunk to %u",
                        current_fps, _samples_per_chunk);
		} else {
			TIME_WORKER TaskTimer("Current framerate = %.1f fps", current_fps).suppressTiming();
		}

        _requested_fps *= 0.9;
        if (_min_fps>_requested_fps) _requested_fps = _min_fps;
    }

    // Reset before next workOne
    Tfr::Cwt::Singleton().wavelet_time_support( 2.8 );

    _last_work_one = now;

    return true;
}


///// PROPERTIES

void Worker::
        todo_list( const Signal::Intervals& v )
{
    {
        QMutexLocker l(&_todo_lock);
        _todo_list = v & Interval(0, _source->number_of_samples());
    }

    //todo_list().print(__FUNCTION__);

    if (v)
        _todo_condition.wakeAll();
}


Signal::Intervals Worker::
        todo_list()
{
    QMutexLocker l(&_todo_lock);
    Signal::Intervals c = _todo_list;
    _cheat_work.clear();
    if ( 1 >= Tfr::Cwt::Singleton().wavelet_time_support() )
        c -= _cheat_work;
    else
        _cheat_work.clear();
    return c;
}


Signal::pOperation Worker::
        source() const
{
    return _source;
}


void Worker::
        source(Signal::pOperation value)
{
    _source = value;
    if (_source)
        _min_samples_per_chunk = Tfr::Cwt::Singleton().next_good_size( 1, _source->sample_rate());
    else
        _min_samples_per_chunk = 1;
    _max_samples_per_chunk = (unsigned)-1;
    invalidate_post_sink( Signal::Interval(0, value->number_of_samples() ));

    emit source_changed();
}


void Worker::
        appendOperation(Signal::pOperation s)
{
    s->source( _source );
    Signal::Intervals still_zeros;
    if (_source)
        still_zeros = _source->zeroed_samples();
    still_zeros &= s->zeroed_samples();
    _source = s;
    invalidate_post_sink( s->affected_samples() - still_zeros );

    emit source_changed();
}


unsigned Worker::
        samples_per_chunk() const
{
    return _samples_per_chunk;
}


void Worker::
		samples_per_chunk_hint(unsigned value)
{
    _samples_per_chunk = max(_min_samples_per_chunk, value);
}


float Worker::
        requested_fps() const
{
    return _requested_fps;
}


void Worker::
        requested_fps(float value)
{
    if (_min_fps>value) value=_min_fps;

    if (value>_requested_fps) {
        _requested_fps = value;
        samples_per_chunk_hint(1);
        _max_samples_per_chunk = (unsigned)-1;
        Tfr::Cwt::Singleton().wavelet_time_support( 0.5 );
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
        invalidate_post_sink(Intervals I)
{
    dynamic_cast<OperationCacheLayer*>(_cache.get())->invalidate_samples( I );
    _post_sink.invalidate_samples( I );
    TaskInfo("Worker invalidate %s. Worker tree:\n%s",
             I.toString().c_str(),
             source()?source()->toString().c_str():0);
}


/*PostSink* Worker::
        postSink()
{
    return &_post_sink;
}*/


// TODO remove
/*std::vector<pOperation> Worker::
        callbacks()
{
    QMutexLocker l(&_callbacks_lock);
    std::vector<pOperation> c = _callbacks;
    return c;
}*/


///// PRIVATE


void Worker::
        run()
{
	while (true)
	{
		try {
            while (todo_list())
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
        addCallback( pOperation c )
{
    c->source(source());

    std::vector<pOperation> callbacks = _post_sink.sinks();
    callbacks.push_back( c );
    _post_sink.sinks(callbacks);
}


void Worker::
        removeCallback( pOperation c )
{
    c->source(pOperation());

    QMutexLocker l( &_callbacks_lock );
    std::vector<pOperation> callbacks = _post_sink.sinks();
    callbacks.resize( std::remove( callbacks.begin(), callbacks.end(), c ) - callbacks.begin() );
    _post_sink.sinks(callbacks);
}


pBuffer Worker::
        callCallbacks( Interval I )
{
    _cache->source(source());
    {
        //Signal::Intervals fetched_invalid = source()->fetch_invalid_samples();
        //dynamic_cast<OperationCacheLayer*>(_cache.get())->invalidate_samples( fetched_invalid );
        //_post_sink.invalidate_samples( fetched_invalid );
    }
    _post_sink.source( _cache );
    //_post_sink.source( source() );
    return _post_sink.read( I );
}

} // namespace Signal
