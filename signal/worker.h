#ifndef SIGNALWORKER_H
#define SIGNALWORKER_H

// Signal namespace
#include "intervals.h"
#include "postsink.h"
#include "chain.h"
#include "target.h"

// Boost
#include <boost/noncopyable.hpp>

// Qt
#ifndef SAWE_NO_MUTEX
#include <QMutex>
#include <QThread>
#include <QWaitCondition>
#endif
#include <QObject>


namespace Signal {

/**
Signal::Worker is a queue of ranges, that might be worked off by one or
multiple threads (and hence one or multiple GPUs). The calling thread
calls Worker::workOne as a blocking operation performing the following
steps:

1. Ask pOperation Signal::Worker::_source for a pBuffer over some samples.
2. The pBuffer is sent to the callback(s).

--- Design motivation for Signal::Worker ---
Heightmap rendering and Playback both needs to read from Signal::Source::read.
But neither are allowed to do so directly because 'read' is a blocking
operation.

At a first glance Playback could afford read as a blocking operation as
Playback is run asynchronously anyway in a separate thread. But it is
impossible to guarantee that 'read' is fast enough (i.e. returns more than
"Source::sample_rate" samples per second). Also cuda kernel invocations
are also blocking, meaning that the entire operating system user interface
will be locked while the kernel executes. These issues call for an
asynchronous solution that makes sure that kernel invocations are fast
enough for a responsive UI, yet big enough for efficient computations.

--- Solution ---
Signal::Source::read is called by an asynchronous worker thread instead.
Playback is notified via Signal::Sink::put( pBuffer, pOperation ) as a callback
function by registering itself as a Signal::Sink in tfr-worker. Tr::Waveform
is notified by the same callback. Heightmap rendering is also notified by the
same callback and will have to compute the cwt of the Buffer, unless
dynamic_cast<CwtFilter*>(source.get())==true in which case Heightmap
rendering can reuse the cwt previously computed in the CwtFilter.

These callbacks are invoked by different threads if there are multiple GPUs
available, one thread per GPU and CUDA context. The callback may use any CUDA
objects reachable through Buffer or Source for further computations. If such
computations are heavy they will affect the total time for each loop iteration
in Signal::Worker and thus the size of chunk reads will be adjusted to reach
the desired framerate. Also note that OpenGL rendering requires data to be
transfered from CUDA via host memory (CPU RAM) unless the main thread doing
OpenGL rendering is the same as the thread invoking the callback. Use
Gpumisc::ThreadChecker or QThread to decide if data needs to be moved over to
the CPU.

--- Example ---
This is an example tree of signal operations. Rendering is placed at the first
none-GroupOperation, a GroupOperation doesn't affect the flow but merely keeps
a couple of operations together for more meaningful rendering. A MergeOperation
is a GroupOperation which starts with a pure source (not an operation on a
previous source) and then superimposes the previous source with the source
created by the GroupOperation.

1/ GroupOperation: MergeOperation Root
9 Operation remove section [Rendering]
8 Operation: Amplitude*0.5
7 CwtFilter: remove ellipse
6 GroupOperation: move selection in frequency
  2 Operation: merge extracted selection at new location in time
  2 CwtFilter: resample at new frequency
  1 CwtFilter: extract selection
5 GroupOperation: move selection in time
  2 Operation: merge extracted selection at new location in time
  1 CwtFilter: extract selection
4 Operation: remove section
3 Operation: remove section
2 GroupOperation: move section
  3 Operation: remove section
  2 GroupOperation: copy section
    1 Operation: merge source
  1 Operation: insert silence
1 Source: Recording

The pOperation for Signal::Worker can be set to any pOperation and results are fed to
the callbacks when they are ready. Note that if the source is changed while
Playback is active, sound will be heard from the new source at the same
location in time. Playback will not remember its entire previous tree.

If there is an underfed playback (bool Playback::is_underfed()), Playback is
prioritized and sets the todo_list in Signal::Worker. Otherwise; if it is not
underfed, some rendering can be done and Heightmap can set the todo_list
instead. It is up to the global rendering loop to determine which has higher
priority.

If one of the receiving sinks is underfed, worker will not take smaller
chunks and will not skip working even if the requested frame rate is
higher than the actual achieved framerate.
  */
class Worker
#ifndef SAWE_NO_MUTEX
    : public QThread
#endif
{
public:
    /**
      It is allowed to create Worker with a null target. But the instance may
      not be used until a target is set.
      */
    Worker(pTarget t);
    ~Worker();

    /**
      workOne is called once each frame. workOne times itself and adjusts _samples_per_chunk such that
      it will approximately take more than 10 ms but less than 40 ms. However, _samples_per_chunk will
      always be greater than 2.5*_wavelet_std_samples.

      @return true if some work was done and false otherwise
      */
    bool workOne(bool skip_if_low_fps = true);

    /**
      work_chunks is incremented by one each time workOne is invoked.
      */
    unsigned work_chunks;

    /**
      worked_samples is incremented each time workOne is invoked.
      */
    Intervals worked_samples;

    /**
      Returns the previous Interval that workOne requested
      */
    Interval latest_request;

    /**
      Returns the previous Interval that workOne requested
      */
    Interval latest_result;

    /**
      Returns the last thing returned from fetch todo list.
      */
    Intervals todo_list();

    /**
      This property states which regions that are more important. It should be equivalent to the camera position
      in the t-axis for rendering, and equal to 0 for playback.
      @see todo_list
      */
    float center;


    /**
      The current working point.
      */
    pOperation          source() const;
    Signal::pTarget     target() const;

    /**
      Setting a new target also sets a new todo list.
      */
    void                target(Signal::pTarget target);

    /**
      Get number of samples computed for each iteration.
      */
    unsigned            samples_per_chunk() const;

	/**
	  Suggest number of samples to compute for each iteration.
	*/
    void				samples_per_chunk_hint(unsigned);

    void                nextFrame();
    float               get_current_fps() const;
    float               get_min_fps() const;

    /**
      Get/set requested number of frames per second.
      */
    float               requested_fps() const;
    void                requested_fps(float);

    /**
      source()->number_of_samples() may change during a workOne() call. This
      method will always return the same value during the entire lifespan of
      workOne().
      */
    Signal::IntervalType number_of_samples() const { return _number_of_samples; }
    float length() const { return number_of_samples()/source()->sample_rate(); }

        /**
	  Throws an std::exception if one has been caught by run()
	  */
	void				checkForErrors();


private:
    /**
      The InvalidSamplesDescriptors describe the regions that need to be recomputed. The todo_list
      is rebuilt each time a new region is requested. It is worked off in a outward direction
      from the variable center.
      */
    Intervals fetch_todo_list();

#ifndef SAWE_NO_MUTEX
    /**
      Runs the worker thread.
      */
    virtual void run();
#endif

    //void todo_list( const Intervals& v );

    /**
      Self explanatory.
      */
    pBuffer callCallbacks( Interval i );


    /**
      Number of samples, updated from source when calling workOne().
      */
    Signal::IntervalType _number_of_samples;

    /**
      Adjusting chunk size based on fps.
      */
    boost::posix_time::ptime _last_work_one;

#ifndef SAWE_NO_MUTEX
    /**
      Thread safety for addCallback, removeCallback and callCallbacks.
      */
    QMutex _callbacks_lock;
#endif


#ifndef SAWE_NO_MUTEX
    /**
      Thread safety for _todo_list.
      */
    QMutex _todo_lock;
    QWaitCondition _todo_condition;
#endif

    Signal::pTarget _target;

    /**
      @see todo_list
      */
    Intervals _todo_list, _cheat_work;

    /**
      samples_per_chunk is optimized for optimal cwt speed while still keeping the user interface responsive.
      Default value: samples_per_chunk=4096.
      @see workOne
      */
    unsigned  _samples_per_chunk;
    unsigned  _max_samples_per_chunk;
    unsigned  _min_samples_per_chunk;

    /**
      _samples_per_chunk is adjusted up and down to reach this given framerate. Default value: requested_fps=30.
      */
    float _requested_fps;

    /**
      lowest fps allowed, defaults to 0.5
      */
    float _min_fps;


    /**
      highest measured fps when actually computing something. If requested fps is higher workOne doesn't do anything.
      _highest_fps is guaranteed to be >= _min_fps.
    */
    float _highest_fps;

    /**
      */
    float current_fps;

    /**
      If the worker shouldn't do any work for any target set target to pTarget(). Worker will however keep a
      reference so that source() can be provided.
      */
    bool _disabled;

	/**
      Worker::run is intended to be executed by a separate worker thread. To
      simplify error handling in the GUI thread exceptions are caught by
      Worker::run and stored. A client may poll with: caught_exception.what()
      */
    std::runtime_error _caught_exception;
	std::invalid_argument _caught_invalid_argument;
};
// typedef boost::shared_ptr<Worker> pWorker;


} // namespace Signal

#endif // SIGNALWORKER_H
