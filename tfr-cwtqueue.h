#ifndef TFRCWTQUEUE_H
#define TFRCWTQUEUE_H

#include <QMutex>
#include <boost/shared_ptr.hpp>

namespace Tfr {

/**
   @see CwtQueue
  */
class ChunkCompleteCallback {
public:
    ChunkCompleteCallback( CwtQueue* p ) :_q(p) { _q->addCallback( this ); }
    ~ChunkCompleteCallback( ) { _q->removeCallback( this ); }

    virtual void chunkComplete( Tfr::pChunk ) = 0;

private:
    class CwtQueue* _q;
};

/**

Playback: Same as active rendering source
Rendering source: 6

Rendering always has a Source* as source, the default behaviour is to compute the Cwt from this source.
However, if dynamic_cast<FilterOperation*>(_source) rendering can be done directly from the Cwt requested by
the FilterOperation instead.
Playback has a Source* as source and

There is only one global pCwtQueue. Playback plays from the same queue as is being rendered.

If there is an underfed playback (bool Playback::is_underfed()), Playback is prioritized and sets the todo_list.
Otherwise; if it is not underfed, some rendering can be done and Heightmap can set the todo_list instead.



6 Head: Operation remove section
5 FilterOperation: Remove ellips
4 Operation: remove section
3 Operation: remove section
2 Operation: move section:
  2 Operation: remove section
  1 Operation: copy section
1 Source: Recording

  */


/**
   CwtQueue
        is a queue of ranges, might be worked off by one or multiple GPUs.

        1. ask pSource for a Buffer over those samples
        2. The retrieved Signal::Buffer is sent to Tfr::Cwt,
        (3. The Tfr::Chunk is sent to this->operation (which is likely is an operationlist).)
        4. The Tfr::Chunk is sent to the callback, there might be multiple callbacks on each queue.

        CwtCompleteCallback takes a CwtQueue as constructing argument and can thus access the CwtQueue mutex
        and be more threadsafe. CwtQueue aquires the mutex when calling CwtCompleteCallback, and CwtCompleteCallback
        aquires the mutex upon destruction, where it also removes itself from CwtQueue.
*/
class CwtQueue
{
public:
    CwtQueue(pSource source);

    /**
      workOne is called once each frame. workOne times itself and adjusts _samples_per_chunk such that
      it will approximately take more than 10 ms but less than 40 ms. However, _samples_per_chunk will
      always be greater than 2.5*_wavelet_std_samples.
      */
    void workOne();

    /**
      The InvalidSamplesDescriptors describe the regions that need to be recomputed. The todo_list
      is rebuilt each time a new region is requested. It is worked off in a outward direction
      from the variable center.
      */
    InvalidSamplesDescriptor todo_list;

    /**
      This property states which regions that are more important. It should be equivalent to the camera position
      in the t-axis for rendering, and equal to 0 for playback.
      @see todo_list
      */
    float center;

    /**
      wavelet_std_t is the size of overlapping in the windowed cwt, measured in time units (i.e seconds).
      */
    float     wavelet_std_t() const { return _wavelet_std_samples/(float)_source->sample_rate(); }
    void      wavelet_std_t( float );

    /**
      Get the data source for this worker queue.
      */
    Signal::pSource source() const { return _source; }

    /**
      This object controls all properties for the continious wavelet transform.
      */
    Cwt cwt;
private:
    friend class ChunkCompleteCallback;

    /**
      A ChunkCompleteCallback adds itself to a cwtqueue.
      */
    addCallback( ChunkCompleteCallback* c );
    /**
      A ChunkCompleteCallback removes itself from a cwtqueue.
      */
    removeCallback( ChunkCompleteCallback* c );

    /**
      Self explanatory.
      */
    callCallbacks( Tfr::pChunk );

    /**
      All callbacks in this list are called once for each call of workOne().
      */
    std::list<ChunkCompleteCallback*> callbacks;

    /**
      Thread safety for addCallback, removeCallback and callCallbacks.
      */
    QMutex lock;

    /**
      @see source
      */
    Signal::pSource _source;

    /**
      wavelet_std_t measured in samples instead of seconds. Default value: wavelet_std_samples=0.03.
      @see wavelet_std_t
      */
    unsigned  _wavelet_std_samples;

    /**
      samples_per_chunk is optimized for optimal cwt speed while still keeping the user interface responsive.
      Default value: samples_per_chunk=4096.
      @see workOne
      */
    unsigned  _samples_per_chunk;
};
boost::shared_ptr<CwtQueue> pCwtQueue;

} // namespace Tfr

#endif // TFRCWTQUEUE_H
