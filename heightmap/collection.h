#ifndef HEIGHTMAPCOLLECTION_H
#define HEIGHTMAPCOLLECTION_H

// Heightmap namespace
#include "reference.h"
#include "glblock.h"
#include "block.cu.h"

// Sonic AWE
#include "signal/intervals.h"
#include "signal/operation.h"
#include "tfr/chunk.h"
#include "tfr/transform.h"

// boost
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>

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

namespace Heightmap {


// TODO it would probably look awesome if new blocks weren't displayed
// instantaneously but rather faded in from 0 or from their previous value.
// This method could be used to slide between the images of two different
// signals or channels as well. This should be implemented by rendering two or
// more separate collections in Heightmap::Renderer. It would fetch Blocks by
// their 'Reference' from the different collections and use a shader to
// transfer results between them.
class Block {
public:
    Block( Reference ref )
        :
        frame_number_last_used(-1),
        ref(ref)
#ifndef SAWE_NO_MUTEX
        ,new_data_available( false )
#endif
    {}

    ~Block();

    // TODO move this value to a complementary class
    unsigned frame_number_last_used;

    // Zoom level for this slot, determines size of elements
    Reference ref;
    pGlBlock glblock;

    typedef boost::shared_ptr<GpuCpuData<float> > pData;

    /**
        TODO test this in a multi gpu environment
        For multi-GPU or (just multithreaded) environments, each GPU-thread have
        its own cuda context and data can't be  transfered between cuda contexts
        without first going to the cpu. Therefore a 'cpu_copy' is kept in CPU
        memory so that the block data is readily available for merging new
        blocks. Only one GPU may access 'cpu_copy' at once. The OpenGL textures
        are updated from cpu_copy whenever new_data_available is set to true.

        For single-GPU environments, 'cpu_copy' is not used.
    */
#ifndef SAWE_NO_MUTEX
    pData cpu_copy;
    bool new_data_available;
    QMutex cpu_copy_mutex;
#endif

    /**
      valid_samples describes the intervals of valid samples contained in this block.
      it is relative to the start of the heightmap, not relative to this block unless this is
      the first block in the heightmap. The samplerate is the sample rate of the full
      resolution signal.
      */
    Signal::Intervals valid_samples;
};
typedef boost::shared_ptr<Block> pBlock;


inline std::size_t hash_value(Reference const& ref)
{
    std::size_t seed = 0;
    boost::hash_combine(seed, ref.log2_samples_size[0]);
    boost::hash_combine(seed, ref.log2_samples_size[1]);
    boost::hash_combine(seed, ref.block_index[0]);
    boost::hash_combine(seed, ref.block_index[1]);
    return seed;
}


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
    virtual void reset();
    virtual bool empty();


    virtual Signal::Intervals invalid_samples();
    virtual void invalidate_samples( const Signal::Intervals& );


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
      next_frame returns that counter. next_frame also calls applyUpdates().
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
      */
    std::vector<pBlock>      getIntersectingBlocks( Signal::Interval I, bool only_visible );


    /**
      Notify the Collection what filter that is currently creating blocks.
      */
    void block_filter(Signal::pOperation filter) { _filter = filter; }
    Signal::pOperation block_filter() { return _filter; }


    /**
      Extract the transform from the current filter.
      */
    Tfr::pTransform transform();


    unsigned long cacheByteSize();
    unsigned    cacheCount();
    void        printCacheSize();
    void        gc();
    void        discardOutside(Signal::Interval I);


    Tfr::FreqAxis display_scale() { return _display_scale; }
    void display_scale(Tfr::FreqAxis a);


    Signal::pOperation target;

    const ThreadChecker& constructor_thread() const { return _constructor_thread; }

private:
    // TODO remove friends
    //friend class BlockFilter;
    friend class CwtToBlock;
    friend class StftToBlock;

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

    Signal::pOperation _filter;

    /**
      Heightmap blocks are rather agnostic to FreqAxis.
      */
    Tfr::FreqAxis _display_scale;


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
    QMutex _cache_mutex; // Mutex for _cache and _recent
#endif
    cache_t _cache;
    recent_t _recent; // Ordered with the most recently accessed blocks first

    // QMutex _updates_mutex;
    // QWaitCondition _updates_condition;
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
      Work of the _updates queue of chunks to merge.
      */
    // void        applyUpdates();


    /**
      TODO comment
      */
    void        mergeStftBlock( Tfr::pChunk stft, pBlock block );


    /**
      Add block information from Cwt transform. Returns whether any information was merged.
      */
    bool        mergeBlock( Block* outBlock, Tfr::Chunk* inChunk, unsigned cuda_stream, bool save_in_prepared_data = false );

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
    Signal::Intervals getInvalid(const Reference& r);
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
