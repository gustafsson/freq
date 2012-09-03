#ifndef HEIGHTMAPCOLLECTION_H
#define HEIGHTMAPCOLLECTION_H

// Heightmap namespace
#include "reference_hash.h"
#include "amplitudeaxis.h"
#include "tfr/freqaxis.h"

// Sonic AWE
#include "signal/intervals.h"
#include "signal/poperation.h"

// gpumisc
#include "ThreadChecker.h"

// boost
#include <boost/unordered_map.hpp>

// Qt
#ifndef SAWE_NO_MUTEX
#include <QMutex>
#include <QWaitCondition>
#include <QThread>
#endif

// std
#include <vector>

/*
TODO: rewrite this section

Data structures
---
The transform is rendered in blocks at different zoomlevels, think google maps.
Each block is queued for computation (instead of being downloaded) when it is
requested for.

These blocks have an arbitrary aspect ratio, unlike map software where the
width is always equal to the height.

After each viewport change the queue of requested blocks are reordered after
"importance" and dependencies.

There is however a constant number of data points in each dimension of a block.
Sizes of such a data point is governed by two integer zoom factors:
(delta t, delta f) = (2^tzoom, 2^fzoom) [s, 1/s]
tzoom = INT_MIN is special and always represents the same sample rate as the
original waveform. For all other tzoom, 2^-tzoom is always less than the
waveform sample rate.


Rendering
---
2D uses anisotropic filtering. Data at different mipmap levels are explicitly
provided from previously computed blocks.

3D by creating a vertex array out of the amplitudes of each block. The blocks
are rendered independently as in the 2D case.

During rendering, non-existing blocks may get requested for. A block request is
almost always successful.


Computation
---
Only the most high resolution block can be computed from sratch. That block has
the same sample rate along the t-axis as the original waveform. Resolution in
frequency is at least as high as some treshold value. Such as 10 scales per
octave for instance.

To compute other blocks, higher resolution blocks are downsampled and stored in
subdivisions of a block with the requested resolution.

Whenever a new block size is requested, it is first approximated by an
interpolation from any available lower resolution block. If no such block is
available it is sufficient to first render it black, then compute a STFT
approximation of the wavelettransform, and then compute the real
wavelettransform.


Filters
---
Filters that extract or alter data are applied when the transform is computed,
and will thus alter the rendering. To render a spectogram without having the
effects of filters visualized an empty filter chain can be passed into
Spectogram. For altering the output refer to transform-inverse.h.

The term scaleogram is not used in the source code, in favor of spectrogram.
*/


namespace Tfr {
    class Chunk;
    class Transform;
}

namespace Heightmap {

class Renderer;
class Block;

typedef boost::shared_ptr<Block> pBlock;


/**
  Signal::Sink::put is used to insert information into this collection.
  getBlock is used to extract blocks for rendering.
  */
class Collection {
public:
    Collection(Signal::pOperation target);
    ~Collection();


    /**
      Releases all GPU resources allocated by Heightmap::Collection.
      */
    void reset();
    bool empty();


    Signal::Intervals invalid_samples();
    void invalidate_samples( const Signal::Intervals& );


    /**
      scales_per_block and samples_per_block are constants deciding how many blocks
      are to be created.
      */
    unsigned    scales_per_block() const { return _scales_per_block; }
    void        scales_per_block(unsigned v);
    unsigned    samples_per_block() const { return _samples_per_block; }
    void        samples_per_block(unsigned v);


    /**
      getBlock increases a counter for each block that hasn't been computed yet.
      next_frame returns that counter.
      */
    unsigned    next_frame();


    /**
      Returns a Reference for a block containing the entire heightmap.
      */
    Reference   entireHeightmap();


    /**
      Get a heightmap block. If the referenced block doesn't exist it is created.

      This method is used by Heightmap::Renderer to get the heightmap data of
      blocks that has been decided for rendering.

      Might return 0 if collections decides that it doesn't want to allocate
      another block.
      */
    pBlock      getBlock( const Reference& ref );


    /**
      Blocks are updated by CwtToBlock and StftToBlock by merging chunks into
      all existing blocks that intersect with the chunk interval.

      This method is called by working threads.
      */
    std::vector<pBlock>      getIntersectingBlocks( const Signal::Intervals& I, bool only_visible );


    /**
      Notify the Collection what filter that is currently creating blocks.
      */
    void block_filter(Signal::pOperation filter) { _filter = filter; }
    Signal::pOperation block_filter() { return _filter; }


    /**
      Extract the transform from the current filter.
      */
    boost::shared_ptr<Tfr::Transform> transform();


    unsigned long cacheByteSize();
    unsigned    cacheCount();
    void        printCacheSize();
    void        gc();
    void        discardOutside(Signal::Interval I);


    Tfr::FreqAxis display_scale() { return _display_scale; }
    void display_scale(Tfr::FreqAxis a);

    Heightmap::AmplitudeAxis amplitude_axis() { return _amplitude_axis; }
    void amplitude_axis(Heightmap::AmplitudeAxis a);

    Signal::pOperation target;

    const ThreadChecker& constructor_thread() const { return _constructor_thread; }

    bool isVisible();
    void setVisible(bool v);

    Renderer* renderer;

private:
    // TODO remove friends
    //friend class BlockFilter;
    friend class CwtToBlock;
    friend class StftToBlock;

    bool
        _is_visible;

    unsigned
        _samples_per_block,
        _scales_per_block,
        _unfinished_count,
        _created_count,
        _frame_counter;

    float
        _prev_length;

    Position
            _max_sample_size;

    // free memory is updated in next_frame()
    size_t
            _free_memory;

    Signal::pOperation _filter;

    /**
      Heightmap blocks are rather agnostic to FreqAxis. But it's needed to create them.
      */
    Tfr::FreqAxis _display_scale;

    /**
      Heightmap blocks are rather agnostic to Heightmap::AmplitudeAxis. But it's needed to create them.
      */
    Heightmap::AmplitudeAxis _amplitude_axis;

    /**
      The cache contains as many blocks as there are space for in the GPU ram.
      If allocation of a new block fails to be allocated
            1) all unused blocks are freed.
            2) if no unused blocks are found and _cache is non-empty, the entire _cache is cleared.
            3) if _cache is empty, Sonic AWE is terminated with an OpenGL or Cuda error.
      */

    typedef boost::unordered_map<Reference, pBlock> cache_t;
    typedef std::list<pBlock> recent_t;

#ifndef SAWE_NO_MUTEX
    QMutex _cache_mutex; /// Mutex for _cache and _recent, see getIntersectingBlocks
#endif
    cache_t _cache;
    recent_t _recent; /// Ordered with the most recently accessed blocks first

    ThreadChecker _constructor_thread;

    /**
      Attempts to allocate a new block.
      */
    pBlock      attempt( const Reference& ref );

    /**
      Creates a new block.
      */
    pBlock      createBlock( const Reference& ref );

    /**
      Compoute a short-time Fourier transform (stft). Usefull for filling new
      blocks with data really fast.
      */
    void        fillBlock( pBlock block, const Signal::Intervals& to_update );


    /**
      TODO comment
      */
    void        mergeStftBlock( boost::shared_ptr<Tfr::Chunk> stft, pBlock block );


    /**
      Add block information from another block. Returns whether any information was merged.
      */
    bool        mergeBlock( pBlock outBlock, pBlock inBlock, unsigned cuda_stream );


    /**
      If 'r' exists in _cache, update its last_frame_used so that it wont yet be freed.
      */
    void        poke(const Reference& r);

    /**
      Try to create 'r' and return its invalid samples if it was created.
      */
    Signal::Intervals getInvalid(const Reference& r) const;
};

} // namespace Heightmap

#endif // HEIGHTMAPCOLLECTION_H

/*
git daemon --verbose --base-path=/home/johan/git
sudo -H -u git gitosis-init < ~/id_rsa.pub
stat /home/git/repositories/gitosis-admin.git/hooks/post-update
sudo chmod 755 /home/git/repositories/gitosis-admin.git/hooks/post-update
git clone git://jovia.net/sonicawe.git
sudo -u git git-daemon --base-path=/home/johan

git daemon --verbose --export-all /home/johan/dev/sonic-gpu/sonicawe
                               --interpolated-path=/pub/%IP/%D
                               /pub/192.168.1.200/software
                               /pub/10.10.220.23/software
*/
