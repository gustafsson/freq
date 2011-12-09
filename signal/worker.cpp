#include "worker.h"

// signal
#include "intervals.h"
#include "postsink.h"
#include "operationcache.h"

// Sonic AWE
#include "tfr/cwt.h" // hack to make chunk sizes adapt to computer speed and memory, should use transform instead
#include "sawe/application.h"

// gpumisc
#include <computationkernel.h>
#include <demangle.h>

// qt
#include <QTime>
#ifndef SAWE_NO_MUTEX
#include <QMutexLocker>
#endif

// boost
#include <boost/foreach.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

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
    latest_request(0,0),
    latest_result(0,0),
    _number_of_samples(0),
    _length(0.f),
    _last_work_one(boost::date_time::not_a_date_time),
    _samples_per_chunk( 1 ),
    _max_samples_per_chunk( (unsigned)-1 ),
    _requested_fps( 20 ),
    _requested_cheat_fps( 20 ),
    _min_fps( 2 ),  // Always request at least 2 fps.
                    // At least 1 is required to minimize the risk that CUDA
                    // will screw up playback by blocking the OS and causing audio
                    // starvation and kernel timeouts.
                    // But a framerate of 1 makes it barely usable. 2 is also
                    // questionable, but ok since we get so much gain from
                    // large chunks.
    current_fps( 1 ),
    _disabled( false ),
    _caught_exception( "" ),
    _caught_invalid_argument("")
{
    _highest_fps = _min_fps;

    if (t)
        target( t );
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
        TaskInfo("Worker::workOne. Can't do any work without a target");
        return false;
    }

    updateLength();

    //Signal::Intervals todo_list = this->fetch_todo_list();
    Signal::Intervals todo_list = _todo_list;// this->fetch_todo_list();
    unsigned center_sample = source()->sample_rate() * center;

    Signal::Intervals centerinterval(center_sample, center_sample+1);
    centerinterval = centerinterval.enlarge(_samples_per_chunk*4);

    if (is_cheating() && (todo_list & centerinterval).count()>2*_samples_per_chunk)
        _requested_fps = _requested_cheat_fps;

    if (_target->post_sink()->isUnderfed())
        _requested_fps = _min_fps;

    if (skip_if_low_fps)
        if (_requested_fps>_highest_fps)
            return false;

    if (todo_list.empty())
        return false;

    if (TESTING_PERFORMANCE) _samples_per_chunk = _max_samples_per_chunk;
    work_chunks++;

    Interval interval = todo_list.fetchInterval( _samples_per_chunk, center_sample );
    if (is_cheating() && interval.last > _number_of_samples)
    {
        interval.first = _number_of_samples < _samples_per_chunk ? 0 : _number_of_samples - _samples_per_chunk;
        interval.last = _number_of_samples;
        if ( 0 == interval.count())
            interval.last = interval.first + 1;
    }
    latest_request = interval;

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
                    "Processing %s. From %s at %u, with stepsize %u",
                    interval.toString().c_str(),
                    todo_list.toString().c_str(),
                    center_sample,
                    _samples_per_chunk));
    }

    _cheat_work |= interval;

    pBuffer b;

    try
    {
        ComputationCheckError();

        b = callCallbacks( interval );

        latest_result = b->getInterval();
        worked_samples |= latest_result;

        _samples_per_chunk = b->number_of_samples();

        WORKER_INFO {
            tt->info("Worker got %s x %d, [%g, %g) s. %g or %g x realtime",
                b->getInterval().toString().c_str(), b->channels(),
                b->start(), b->start()+b->length(),
                interval.count()/tt->elapsedTime()/b->sample_rate,
                b->length()/tt->elapsedTime());
        }

        ComputationCheckError();
#ifdef USE_CUDA
    } catch (const CudaException& e ) {
        unsigned min_samples_per_chunk = _target->next_good_size( 1 );

        bool trySmallerChunk = false;
        if (const CufftException* cuffte = dynamic_cast<const CufftException*>(&e))
        {
            trySmallerChunk |= cuffte->getCufftError() == CUFFT_EXEC_FAILED;
            trySmallerChunk |= cuffte->getCufftError() == CUFFT_ALLOC_FAILED;
        }
        else
        {
            trySmallerChunk |= e.getCudaError() == cudaErrorMemoryAllocation;
        }

        if (trySmallerChunk && min_samples_per_chunk<_samples_per_chunk) {
            TaskInfo ti("Worker caught cudaErrorMemoryAllocation\n%s",  e.what());
            cudaGetLastError(); // consume error
            TaskInfo("Samples per chunk was %u", _samples_per_chunk);
            TaskInfo("Max samples per chunk was %u", _max_samples_per_chunk);
            TaskInfo("Scales per octave is %g", Tfr::Cwt::Singleton().scales_per_octave() );

            _samples_per_chunk = _target->prev_good_size( _samples_per_chunk );

            if (_max_samples_per_chunk == _samples_per_chunk)
                throw;

            _max_samples_per_chunk = _samples_per_chunk;

            TaskInfo("Setting max samples per chunk to %u", _samples_per_chunk);

            size_t free=0, total=0;
            cudaMemGetInfo(&free, &total);
            TaskInfo("Cuda RAM size %g MB (of which %g MB are currently available)",
                     total/1024.f/1024, free/1024.f/1024);

        } else {
            throw;
        }
//            TaskInfo("Worker caught CudaException:\n%s", e.what());
//            throw;
//        }
#endif
    } catch (const exception& e) {
        TaskInfo("Worker caught exception type %s:\n%s",
                  vartype(e).c_str(), e.what());
        throw;
    } catch (...) {
        TaskInfo("Worker caught unknown exception.");
        throw;
    }

    if (b && !_last_work_one.is_not_a_date_time()) if (!TESTING_PERFORMANCE) {
        unsigned prev_samples_per_chunk = _samples_per_chunk;
        if (current_fps < _requested_fps)
        {
            _samples_per_chunk = _target->prev_good_size( _samples_per_chunk );
            if (_samples_per_chunk == prev_samples_per_chunk)
                _highest_fps = current_fps;

            WORKER_INFO TaskInfo(
                    "Low framerate (%.1f fps). Decreased samples per chunk to %u",
                    current_fps, _samples_per_chunk);
        }
        else if (current_fps > 1.2f*_requested_fps)
        {
            unsigned new_samples_per_chunk = _target->next_good_size( _samples_per_chunk );
            float p = new_samples_per_chunk / (float)prev_samples_per_chunk;
            if ( current_fps/p >= _requested_fps )
            {
                _samples_per_chunk = new_samples_per_chunk;
                if (_samples_per_chunk>_max_samples_per_chunk)
                    _samples_per_chunk=_max_samples_per_chunk;
                else
                    WORKER_INFO TaskInfo(
                            "High framerate (%.1f fps). Increased samples per chunk to %u",
                            current_fps, _samples_per_chunk);
            }
		} else {
			TIME_WORKER TaskTimer("Current framerate = %.1f fps", current_fps).suppressTiming();
		}

        if (_highest_fps < current_fps)
            _highest_fps = current_fps;
        if (_highest_fps < _min_fps)
            _highest_fps = _min_fps;
    }

    current_fps = -1;

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
    WORKER_INFO TaskTimer tt("Fetching todo");
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_todo_lock);
#endif
    Signal::Intervals todoinv = _target->post_sink()->invalid_samples();
    todoinv &= _target->post_sink()->getInterval();

    Signal::Intervals c = todoinv;
    c -= _cheat_work;

    bool is_cheating = this->is_cheating();
    if (is_cheating && !c)
    {
        // Not cheating anymore as there would have been nothing left to work on
        WORKER_INFO TaskInfo("Restoring time suppor");
        Tfr::Cwt::Singleton().wavelet_time_support( Tfr::Cwt::Singleton().wavelet_default_time_support() );
    }

    if (!is_cheating || !c)
    {
        _cheat_work.clear();
        c = todoinv;
    }


    BOOST_FOREACH(Signal::Interval const &b, c)
    {
        if (b.count() == 0) {
            TaskInfo ti("Worker interval debug. ");
            ti.tt().getStream() << "\nInvalid samples: " << _target->post_sink()->invalid_samples() << "\nPS interval: "
                    << _target->post_sink()->getInterval() << "\nCheat interval:" << _cheat_work;
        }
    }

    return _todo_list = c;
}


Intervals Worker::
        todo_list()
{
    return _todo_list;
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
        _todo_list.clear();
        return;
    }

    if (_target != value)
        _highest_fps = _min_fps;

    _target = value;

    updateLength();

    if (!_target->allow_cheat_resolution())
        Tfr::Cwt::Singleton().wavelet_time_support( Tfr::Cwt::Singleton().wavelet_default_time_support() );

    fetch_todo_list();
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


float Worker::
        get_current_fps() const
{
    return current_fps;
}


float Worker::
        min_fps() const
{
    return _min_fps;
}


void Worker::
        min_fps(float f)
{
    _min_fps = f;

    if (_highest_fps < _min_fps)
        _highest_fps = _min_fps;
}


void Worker::
        nextFrame()
{
    _requested_fps *= 0.8f;
    if (is_cheating() && _requested_fps < _requested_cheat_fps)
        _requested_fps = _requested_cheat_fps;
    if (_requested_fps < _min_fps)
        _requested_fps = _min_fps;

    ptime now = microsec_clock::local_time();
    if (!_last_work_one.is_not_a_date_time())
    {
        time_duration diff = now - _last_work_one;
        if (current_fps<0)
            current_fps = 1000000.0/diff.total_microseconds();
        else
            current_fps = 0;
    }
    _last_work_one = now;

    Tfr::Cwt::Singleton().wavelet_time_support( Tfr::Cwt::Singleton().wavelet_default_time_support() );
}


float Worker::
        requested_fps() const
{
    return _requested_fps;
}


void Worker::
        requested_fps(float value, float cheat_value)
{
    if (_min_fps>value) value=_min_fps;
    if (_min_fps>cheat_value) cheat_value=_min_fps;


    if (value>_requested_fps) {
        _max_samples_per_chunk = (unsigned)-1;
        if (_target->allow_cheat_resolution())
        {
            Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
            float fs = target()->source()->sample_rate();
            float fast_support_samples = 0.01f *fs;
            float fast_support = fast_support_samples / cwt.morlet_sigma_samples( fs, cwt.wanted_min_hz() );
            if (fast_support < cwt.wavelet_time_support())
                cwt.wavelet_fast_time_support( fast_support );
        }
    }


    _requested_cheat_fps = cheat_value;
    if (_requested_cheat_fps > _requested_fps)
        _samples_per_chunk = _target->next_good_size(1);


    if (value>_requested_fps)
        _requested_fps = value;
}


bool Worker::
        is_cheating()
{
    return Tfr::Cwt::Singleton().wavelet_time_support() < Tfr::Cwt::Singleton().wavelet_default_time_support();
}


void Worker::
        updateLength()
{
    _number_of_samples = source()->number_of_samples();
    _length = source()->length();
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
