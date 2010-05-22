#ifndef SIGNALWORKER_H
#define SIGNALWORKER_H

#include "signal-sink.h"
#include "signal-samplesintervaldescriptor.h"
#include <boost/noncopyable.hpp>
#include <QMutex>

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

1. Ask pSource Signal::Worker::_source for a pBuffer over some samples.
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
Playback is notified via Signal::Sink::put( pBuffer, pSource ) as a callback
function by registering itself as a Signal::Sink in tfr-worker. Tr::Waveform
is notified by the same callback. Heightmap rendering is also notified by the
same callback and will have to compute the cwt of the Buffer, unless
dynamic_cast<FilterOperation*>(source.get())==true in which case Heightmap
rendering can reuse the cwt previously computed in the FilterOperation.

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
7 FilterOperation: remove ellips
6 GroupOperation: move selection in frequency
  2 Operation: merge extracted selection at new location in time
  2 FilterOperation: resample at new frequency
  1 FilterOperation: extract selection
5 GroupOperation: move selection in time
  2 Operation: merge extracted selection at new location in time
  1 FilterOperation: extract selection
4 Operation: remove section
3 Operation: remove section
2 GroupOperation: move section
  3 Operation: remove section
  2 GroupOperation: copy section
    1 Operation: merge source
  1 Operation: insert silence
1 Source: Recording

The pSource for Signal::Worker can be set to any pSource and results are fed to
the callbacks when they are ready. Note that if the source is changed while
Playback is active, sound will be heard from the new source at the same
location in time. Playback will not remember its entire previous tree.

If there is an underfed playback (bool Playback::is_underfed()), Playback is
prioritized and sets the todo_list in Signal::Worker. Otherwise; if it is not
underfed, some rendering can be done and Heightmap can set the todo_list
instead. It is up to the global rendering loop to determine which has higher
priority.
  */
class Worker
{
public:
    Worker(pSource source);

    /**
      workOne is called once each frame. workOne times itself and adjusts _samples_per_chunk such that
      it will approximately take more than 10 ms but less than 40 ms. However, _samples_per_chunk will
      always be greater than 2.5*_wavelet_std_samples.

      @param middle_chunk tells from were work should be commenced, defaults to 0.
      @return true if some work was done and false otherwise
      */
    bool workOne( unsigned middle_chunk=0 );

    /**
      The InvalidSamplesDescriptors describe the regions that need to be recomputed. The todo_list
      is rebuilt each time a new region is requested. It is worked off in a outward direction
      from the variable center.
      */
    SamplesIntervalDescriptor todo_list;

    /**
      This property states which regions that are more important. It should be equivalent to the camera position
      in the t-axis for rendering, and equal to 0 for playback.
      @see todo_list
      */
    float center;

    /**
      Get/set the data source for this worker.
      */
    Signal::pSource     source() const;
    void                source(Signal::pSource s);

    /**
      Get number of samples computed for each iteration.
      */
    unsigned            samples_per_chunk() const;

    /**
      Get/set requested number of frames per second.
      */
    unsigned            requested_fps() const;
    void                requested_fps(unsigned);

private:
    friend class WorkerCallback;

    /**
      A ChunkCompleteCallback adds itself to a cwtqueue.
      */
    void addCallback( Sink* c );

    /**
      A ChunkCompleteCallback removes itself from a cwtqueue.
      */
    void removeCallback( Sink* c );

    /**
      Self explanatory.
      */
    void callCallbacks( pBuffer );

    /**
      All callbacks in this list are called once for each call of workOne().
      */
    std::list<Sink*> _callbacks; // TODO use pSink

    /**
      Thread safety for addCallback, removeCallback and callCallbacks.
      */
    QMutex _callbacks_lock;

    /**
      @see source
      */
    Signal::pSource _source;

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
};
typedef boost::shared_ptr<Worker> pWorker;


/**
   @see Worker
  */
class WorkerCallback: boost::noncopyable {
public:
    WorkerCallback( pWorker w, pSink s )
        :   _w(w),
            _s(s)
    {
        _w->addCallback( _s.get() );
    }
    ~WorkerCallback( ) { _w->removeCallback( _s.get() ); }

    pWorker worker() { return _w; }
    pSink sink() { return _s; }

private:
    pWorker _w;
    pSink _s;
};
typedef boost::shared_ptr<WorkerCallback> pWorkerCallback;

} // namespace Signal

#endif // SIGNALWORKER_H
