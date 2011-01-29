#include "worker.h"
#include "intervals.h"
#include "postsink.h"
#include "operationcache.h"

#include "tfr/cwt.h" // hack to make chunk sizes adapt to computer speed and memory
#include "sawe/application.h"

#include <QTime>
#ifndef SAWE_NO_MUTEX
#include <QMutexLocker>
#endif
#include <CudaException.h>
#include <demangle.h>

#define TIME_WORKER
//#define TIME_WORKER if(0)

//#define WORKER_INFO
#define WORKER_INFO if(0)

#define TESTING_PERFORMANCE false

#ifdef max
#undef max
#endif

using namespace std;
using namespace boost::posix_time;

namespace Signal {

Worker::
        Worker(Signal::pOperation s)
:   work_chunks(0),
    _last_work_one(boost::date_time::not_a_date_time),
    _source(s),
    _samples_per_chunk( 1 ),
    _max_samples_per_chunk( (unsigned)-1 ),
    _min_samples_per_chunk( 1 ),
    _requested_fps( 20 ),
    _min_fps( 0.5 ),
    _caught_exception( "" ),
    _caught_invalid_argument("")
{
    _highest_fps = _min_fps;

    if(s) source( s );
    // Could create an first estimate of _samples_per_chunk based on available memory
    // unsigned mem = CudaProperties::getCudaDeviceProp( CudaProperties::getCudaCurrentDevice() ).totalGlobalMem;
    // but 1<< 12 works well on most GPUs
}


Worker::
        ~Worker()
{
    TaskInfo tt(__FUNCTION__);

#ifndef SAWE_NO_MUTEX
    this->quit();
#endif
    todo_list( Intervals() );

    std::vector<pOperation> v;
    _post_sink.sinks( v );
    _post_sink.filter( pOperation() );
    _source.reset();
}

///// OPERATIONS


bool Worker::
        workOne()
{
    _requested_fps *= 0.9;
    if (_requested_fps < _min_fps) 
        _requested_fps = _min_fps;

    if (!_post_sink.isUnderfed() && _requested_fps>_highest_fps)
        return false;

    if (todo_list().empty())
        return false;

    if (TESTING_PERFORMANCE) _samples_per_chunk = _max_samples_per_chunk;
    work_chunks++;

    unsigned center_sample = source()->sample_rate() * center;

    Interval interval = todo_list().getInterval( _samples_per_chunk, center_sample );

    boost::scoped_ptr<TaskTimer> tt;

    TIME_WORKER {
        if (interval.count() == _samples_per_chunk)
            tt.reset( new TaskTimer(
                    "Processing %s. From %s at %u",
                    interval.toString().c_str(),
                    todo_list().toString().c_str(),
                    center_sample));
        else
            tt.reset( new TaskTimer(
                    "Processing %s. From %s at %u, with %u steps",
                    interval.toString().c_str(),
                    todo_list().toString().c_str(),
                    center_sample,
                    _samples_per_chunk));
    }

    _cheat_work |= interval;

    pBuffer b;

    try
    {
        CudaException_CHECK_ERROR();

        b = callCallbacks( interval );

        //float r = 2*Tfr::Cwt::Singleton().wavelet_time_support_samples( b->sample_rate )/b->sample_rate;
        worked_samples |= b->getInterval();
        //work_time += b->length();
        //work_time -= r;

        if (_samples_per_chunk > b->number_of_samples())
            _samples_per_chunk = b->number_of_samples();

        WORKER_INFO {
            tt->info("Worker got %s, [%g, %g) s. %g x realtime",
                b->getInterval().toString().c_str(),
                b->start(), b->start()+b->length(),
                //(b->length()-r)/tt->elapsedTime());
                b->length()/tt->elapsedTime());
        }

        CudaException_CHECK_ERROR();
    } catch (const CudaException& e ) {
        unsigned min_samples_per_chunk = Tfr::Cwt::Singleton().next_good_size(1, source()->sample_rate());
        if (cudaErrorMemoryAllocation == e.getCudaError() && min_samples_per_chunk<_samples_per_chunk) {
            TaskInfo ti("Worker caught cudaErrorMemoryAllocation\n/s",  e.what());
            cudaGetLastError(); // consume error
            TaskInfo("_samples_per_chunk was %u", _samples_per_chunk);
            TaskInfo("_max_samples_per_chunk was %u", _max_samples_per_chunk);
            TaskInfo("scales_per_octave was %g", Tfr::Cwt::Singleton().scales_per_octave() );

/*            while (128 < _samples_per_chunk &&
                   _samples_per_chunk <= Tfr::Cwt::Singleton().prev_good_size(
                    _samples_per_chunk, _source->sample_rate()))
            {
                Tfr::Cwt::Singleton().scales_per_octave( Tfr::Cwt::Singleton().scales_per_octave()*0.99f );
            }
*/
            _samples_per_chunk = Tfr::Cwt::Singleton().prev_good_size(
                    _samples_per_chunk, _source->sample_rate());
            _max_samples_per_chunk = _samples_per_chunk;

            TaskInfo("scales_per_octave is now %g", Tfr::Cwt::Singleton().scales_per_octave() );
            TaskInfo("_samples_per_chunk now is %u", _samples_per_chunk);
            TaskInfo("_max_samples_per_chunk now is %u", _max_samples_per_chunk);
            TaskInfo("Setting max samples per chunk to %u", _samples_per_chunk);

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

        if (_highest_fps < current_fps)
            _highest_fps = current_fps;
    }

    // Reset before next workOne
    //TaskInfo("wavelet_default_time_support = %g", Tfr::Cwt::Singleton().wavelet_default_time_support());
    Tfr::Cwt::Singleton().wavelet_time_support( Tfr::Cwt::Singleton().wavelet_default_time_support() );

    _last_work_one = now;

    return true;
}


///// PROPERTIES

void Worker::
        todo_list( const Signal::Intervals& v )
{
    {
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_todo_lock);
#endif
        _todo_list = v & Interval(0, std::max(1lu, _source->number_of_samples()));
        //_todo_list &= Signal::Intervals(0, 44100*7);

        WORKER_INFO TaskInfo("Worker::todo_list = %s (requested %s)",
                             _todo_list.toString().c_str(), v.toString().c_str());
    }

#ifndef SAWE_NO_MUTEX
    if (v)
        _todo_condition.wakeAll();
#endif
}


Signal::Intervals Worker::
        todo_list()
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_todo_lock);
#endif
    if ( Tfr::Cwt::Singleton().wavelet_time_support() >= Tfr::Cwt::Singleton().wavelet_default_time_support() )
        _cheat_work.clear();

    Signal::Intervals c = _todo_list;
    c -= _cheat_work;

    if (!c)
    {
        Tfr::Cwt::Singleton().wavelet_time_support( Tfr::Cwt::Singleton().wavelet_default_time_support() );
        return _todo_list;
    }

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
    BOOST_ASSERT( value );

    _source = value;
    if (_source)
        _min_samples_per_chunk = Tfr::Cwt::Singleton().next_good_size( 1, _source->sample_rate());
    else
        _min_samples_per_chunk = 1;
    _highest_fps = _min_fps;
    _max_samples_per_chunk = (unsigned)-1;
    invalidate_post_sink( Signal::Interval(0, std::max( 1lu, value->number_of_samples()) ));

    emit source_changed();
}


void Worker::
        appendOperation(Signal::pOperation s)
{
    BOOST_ASSERT( s );

    TaskInfo tt("Worker::appendOperation");

    // Check that this operation is not already in the list. Can't move into
    // composite operations yet as there is no operation iterator implemented.
    unsigned i = 0;
    Signal::pOperation itr = _source;
    while(itr)
    {
        if ( itr == s )
        {
            std::stringstream ss;
            ss << "Worker::appendOperation( " << vartype(s) << ", " << std::hex << s.get() << ") "
                    << "is already in operation chain at postion " << i;
            throw std::invalid_argument( ss.str() );
        }
        itr = itr->source();
        ++i;
    }

    s->source( _source );
    Signal::Intervals still_zeros;
    if (_source)
        still_zeros = _source->zeroed_samples_recursive();
    still_zeros &= s->zeroed_samples_recursive();
    _source = s;
    invalidate_post_sink( s->affected_samples() - still_zeros );

    _min_samples_per_chunk = Tfr::Cwt::Singleton().next_good_size( 1, _source->sample_rate());
    _highest_fps = _min_fps;
    _max_samples_per_chunk = (unsigned)-1;

    _source.reset( new OperationCacheLayer(_source) );

    tt.tt().info("Worker::appendOperation, worker tree:\n%s", _source->toString().c_str());

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
        Tfr::Cwt::Singleton().wavelet_fast_time_support( 0.5 );
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
    I &= Interval(0, source()->number_of_samples() );

    {
        // TODO Is this necessary
        OperationCacheLayer* topcache = dynamic_cast<OperationCacheLayer*>(_source.get());
        if (topcache) topcache->invalidate_samples( I );
    }

    _post_sink.invalidate_samples( I );
    WORKER_INFO TaskInfo("Worker invalidate %s. Worker tree:\n%s",
             I.toString().c_str(),
             source()?source()->toString().c_str():0);
}


///// PRIVATE


#ifndef SAWE_NO_MUTEX
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
#ifndef SAWE_NO_MUTEX
            QMutexLocker l(&_todo_lock);
#endif
			_todo_condition.wait( &_todo_lock );}
		} catch ( const std::exception& x ) {
                        _caught_exception = runtime_error(x.what());
			return;
		}
	}
}
#endif


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

#ifndef SAWE_NO_MUTEX
    QMutexLocker l( &_callbacks_lock );
#endif
    std::vector<pOperation> callbacks = _post_sink.sinks();
    callbacks.resize( std::remove( callbacks.begin(), callbacks.end(), c ) - callbacks.begin() );
    _post_sink.sinks(callbacks);
}


pBuffer Worker::
        callCallbacks( Interval I )
{
    _post_sink.source( source() );
    return _post_sink.read( I );
}

} // namespace Signal
