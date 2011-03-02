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
        Worker(Signal::pTarget t)
:   work_chunks(0),
    _number_of_samples(0),
    _last_work_one(boost::date_time::not_a_date_time),
    _samples_per_chunk( 1 ),
    _max_samples_per_chunk( (unsigned)-1 ),
    _min_samples_per_chunk( 1 ),
    _requested_fps( 20 ),
    _min_fps( 0.5 ),
    _disabled( false ),
    _caught_exception( "" ),
    _caught_invalid_argument("")
{
    _highest_fps = _min_fps;

    if (t) target( t );
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
    //if ( _target )
    //    todo_list( Intervals() );

    _target.reset();
}

///// OPERATIONS


bool Worker::
        workOne( bool skip_if_low_fps )
{
    if (_disabled)
    {
        TaskInfo("Can't do any work without a target");
        return false;
    }

    _number_of_samples = source()->number_of_samples();

    _requested_fps *= 0.9;
    if (_requested_fps < _min_fps) 
        _requested_fps = _min_fps;

    if (!_target->post_sink()->isUnderfed() && skip_if_low_fps && _requested_fps>_highest_fps)
        return false;

    Signal::Intervals todo_list = fetch_todo_list();
    if (todo_list.empty())
        return false;

    if (TESTING_PERFORMANCE) _samples_per_chunk = _max_samples_per_chunk;
    work_chunks++;

    unsigned center_sample = source()->sample_rate() * center;

    Interval interval = todo_list.fetchInterval( _samples_per_chunk, center_sample );

    boost::scoped_ptr<TaskTimer> tt;

    TIME_WORKER {
        if (interval.count() == _samples_per_chunk)
            tt.reset( new TaskTimer(
                    "Processing %s. From %s at %u",
                    interval.toString().c_str(),
                    todo_list.toString().c_str(),
                    center_sample));
        else
            tt.reset( new TaskTimer(
                    "Processing %s. From %s at %u, with %u steps",
                    interval.toString().c_str(),
                    todo_list.toString().c_str(),
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
                    _samples_per_chunk, _target->post_sink()->sample_rate());
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
            TaskInfo("scales_per_octave was %g", Tfr::Cwt::Singleton().scales_per_octave());
            Tfr::Cwt::Singleton().scales_per_octave( Tfr::Cwt::Singleton().scales_per_octave() * .75 );
            TaskInfo("scales_per_octave is %g", Tfr::Cwt::Singleton().scales_per_octave());
        }
//            TaskInfo("Worker caught CudaException:\n%s", e.what());
//            throw;
//        }
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
                    _samples_per_chunk, _target->post_sink()->sample_rate());
            WORKER_INFO TaskInfo(
                    "Low framerate (%.1f fps). Decreased samples per chunk to %u",
                    current_fps, _samples_per_chunk);
        }
        else if (current_fps > 2.5f*_requested_fps)
        {
            _samples_per_chunk = Tfr::Cwt::Singleton().next_good_size(
                    _samples_per_chunk, _target->post_sink()->sample_rate());
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

/*void Worker::
        todo_list( const Signal::Intervals& v )
{
    {
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_todo_lock);
#endif
        BOOST_ASSERT( _target );

        _todo_list = v & Interval(0, _target->post_sink()->number_of_samples());
        //_todo_list &= Signal::Intervals(0, 44100*7);

        WORKER_INFO TaskInfo("Worker::todo_list = %s (requested %s)",
                             _todo_list.toString().c_str(), v.toString().c_str());
    }

#ifndef SAWE_NO_MUTEX
    if (v)
        _todo_condition.wakeAll();
#endif
}*/


Signal::Intervals Worker::
        fetch_todo_list()
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_todo_lock);
#endif
    if ( Tfr::Cwt::Singleton().wavelet_time_support() >= Tfr::Cwt::Singleton().wavelet_default_time_support() )
        _cheat_work.clear();

    Signal::Intervals todoinv = _target->post_sink()->invalid_samples();
    todoinv &= _target->post_sink()->getInterval();
    Signal::Intervals c = todoinv;
    c -= _cheat_work;

    if (!c)
    {
        Tfr::Cwt::Singleton().wavelet_time_support( Tfr::Cwt::Singleton().wavelet_default_time_support() );
        return _previous_todo_list = todoinv;
    }

    return _previous_todo_list = c;
}


Intervals Worker::
        previous_todo_list()
{
    return _previous_todo_list;
}


pOperation Worker::
        source() const
{
    return _target->source();
}


pTarget Worker::
        target() const
{
    return _target;
}


void Worker::
        target(pTarget value)
{
    _disabled = 0 == value;

    if (_disabled)
    {
        _number_of_samples = 0;
        return;
    }

    _target = value;

    if (_min_samples_per_chunk==1)
    {
        _min_samples_per_chunk = Tfr::Cwt::Singleton().next_good_size( 1, _target->post_sink()->sample_rate());
        _max_samples_per_chunk = (unsigned)-1;
    }
    _highest_fps = _min_fps;
    _number_of_samples = _target->post_sink()->number_of_samples();
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



///// PRIVATE


#ifndef SAWE_NO_MUTEX
void Worker::
        run()
{
	while (true)
	{
		try {
            while (fetch_todo_list())
			{
				workOne( false );
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


pBuffer Worker::
        callCallbacks( Interval I )
{
    return _target->read( I );
}

} // namespace Signal
