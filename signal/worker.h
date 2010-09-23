#ifndef SIGNALWORKER_H
#define SIGNALWORKER_H

#include "signal/sink.h"
#include "signal/intervals.h"
#include <boost/noncopyable.hpp>
#include <QMutex>
#include <QThread>
#include <QWaitCondition>

namespace Signal {

/**
WorkerCallback is a Signal::Put that registers itself as a worker callback for an instance of the worker class.
*/
class WorkerCallback; // Implementation below.

/**
Signal::Worker is a queue of ranges, that might be worked off by one or
multiple threads (and hence one or multiple GPUs). The calling thread
calls Worker::workOne as a blocking operation performing the following
steps:

1. Ask pOperation Signal::Worker::_source for a pBuffer over some samples.
2. The pBuffer is sent to the callback(s).

WorkerCallback takes a Worker as constructing argument and can thus add itself
to the Worker by Worker::addCallback which uses a mutex to be more threadsafe.
Worker aquires the mutex when calling WorkerCallback, and WorkerCallback
removes itself upon destruction by calling Worker::addCallback which also
aquires the mutex.

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
7 CwtFilter: remove ellips
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
  */
class Worker:public QThread
{
public:
    Worker(pOperation source=pOperation());
    ~Worker();

    /**
      workOne is called once each frame. workOne times itself and adjusts _samples_per_chunk such that
      it will approximately take more than 10 ms but less than 40 ms. However, _samples_per_chunk will
      always be greater than 2.5*_wavelet_std_samples.

      @return true if some work was done and false otherwise
      */
    bool workOne();

    /**
      work_chunks is incremented by one each time workOne is invoked.
      */
    unsigned work_chunks;

    /**
      work_time is incremented each time workOne is invoked.
      */
    float work_time;

    /**
      The InvalidSamplesDescriptors describe the regions that need to be recomputed. The todo_list
      is rebuilt each time a new region is requested. It is worked off in a outward direction
      from the variable center.
      */
    void todo_list( const Intervals& v );
    Intervals todo_list();

    /**
      This property states which regions that are more important. It should be equivalent to the camera position
      in the t-axis for rendering, and equal to 0 for playback.
      @see todo_list
      */
    float center;

    /**
      Get/set the data source for this worker.
      */
    Signal::pOperation     source() const;
    void                source(Signal::pOperation s);

    /**
      Get number of samples computed for each iteration.
      */
    unsigned            samples_per_chunk() const;

	/**
	  Suggest number of samples to compute for each iteration.
	*/
    void				samples_per_chunk_hint(unsigned);

    /**
      Get/set requested number of frames per second.
      */
    unsigned            requested_fps() const;
    void                requested_fps(unsigned);

	/**
	  Throws an std::exception if one has been caught by run()
	  */
	void				checkForErrors();

    /**
      Get all callbacks that data are sent to after each workOne.
      */
    std::vector<pOperation> callbacks();

private:
    friend class WorkerCallback;

    /**
      Runs the worker thread.
      */
    virtual void run();

    /**
      A WorkerCallback adds itself to a Worker.
    @throws invalid_argument if 'c' is not an instance of Signal::Sink.
      */
    void addCallback( pOperation c );

    /**
      A WorkerCallback removes itself from a Worker.
    @throws invalid_argument if 'c' is not an instance of Signal::Sink.
      */
    void removeCallback( pOperation c );

    /**
      Self explanatory.
      */
    pBuffer callCallbacks( Interval i );

    /**
      All callbacks in this list are called once for each call of workOne().
      */
    std::vector<pOperation> _callbacks;

    /**
      Thread safety for addCallback, removeCallback and callCallbacks.
      */
    QMutex _callbacks_lock;

    /**
      @see source
      */
    Signal::pOperation _source;

    /**
      Thread safety for _todo_list.
      */
    QMutex _todo_lock;
    QWaitCondition _todo_condition;

    /**
      @see todo_list
      */
    Intervals _todo_list;

    /**
      samples_per_chunk is optimized for optimal cwt speed while still keeping the user interface responsive.
      Default value: samples_per_chunk=4096.
      @see workOne
      */
    unsigned  _samples_per_chunk;
    unsigned  _max_samples_per_chunk;

    /**
      _samples_per_chunk is adjusted up and down to reach this given framerate. Default value: requested_fps=30.
      */
    unsigned _requested_fps;

	/**
      Worker::run is intended to be executed by a separate worker thread. To
      simplify error handling in the GUI thread exceptions are caught by
      Worker::run and stored. A client may poll with: caught_exception.what()
      */
    std::runtime_error _caught_exception;
	std::invalid_argument _caught_invalid_argument;
};
// typedef boost::shared_ptr<Worker> pWorker;


/**
  TODO this functionality is build upon the purpose of PostSink that is used by Worker. But not in worker itself.

  And also, shouldn't be used like this.

  Hmm, this class is probably not needed at all. But some more refactoring will have to be done.
   @see Worker
  */
class WorkerCallback: boost::noncopyable {
public:
    WorkerCallback( Worker* w, pOperation s )
        :   _w(w),
            _s(s)
    {
        _w->addCallback( _s );
    }
    ~WorkerCallback( ) { _w->removeCallback( _s ); }

    Worker* worker() { return _w; }
    pOperation sink() { return _s; }

private:
    Worker* _w;
    pOperation _s;
};
typedef boost::shared_ptr<WorkerCallback> pWorkerCallback;

} // namespace Signal

#endif // SIGNALWORKER_H
