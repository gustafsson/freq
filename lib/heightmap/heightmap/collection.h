#ifndef HEIGHTMAPCOLLECTION_H
#define HEIGHTMAPCOLLECTION_H

// Heightmap namespace
#include "amplitudeaxis.h"
#include "freqaxis.h"
#include "position.h"
#include "blockcache.h"
#include "render/blocktextures.h"
#include "render/renderset.h"

// Sonic AWE
#include "signal/intervals.h"

// gpumisc
#include "ThreadChecker.h"
#include "deprecated.h"
#include "shared_state.h"

// std
#include <vector>

/*
TODO: rewrite this section

Data structures
---
The heightmap data (tfr transform) is rendered in blocks at different
zoomlevels, think digital map tools. Each block is queued for computation
(instead of being downloaded) when it is requested for.

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

namespace Render {
class Renderer;
}

namespace BlockManagement {
class BlockFactory;
class BlockInitializer;
}

class Block;
class BlockData;

typedef boost::shared_ptr<Block> pBlock;


/**
  Signal::Sink::put is used to insert information into this collection.
  getBlock is used to extract blocks for rendering.
  */
class Collection {
public:
    typedef shared_state<Collection> ptr;

    Collection(BlockLayout, VisualizationParams::const_ptr);
    ~Collection();


    /**
      Releases all GPU resources allocated by Heightmap::Collection.
      */
    void clear();


    Signal::Intervals needed_samples() const;
    Signal::Intervals recently_created();
    Signal::Intervals missing_data();


    /**
      next_frame garbage collects blocks that have been deleted since the last call.
      increments frame_number()
      */
    void     next_frame();


    /**
     * @brief frame_number is the current frame number. Wrapping around at frame
     * number 1 << 32 is assumed to be not an issue.
     */
    unsigned frame_number() const;


    /**
      Returns a Reference for a block containing the entire heightmap.
      */
    Reference   entireHeightmap() const;


    pBlock      getBlock( const Reference& ref );
    void        createMissingBlocks(const Render::RenderSet::references_t& R);
    int         runGarbageCollection( bool aggressive=false );


    unsigned long cacheByteSize() const;
    unsigned    cacheCount() const;
    void        printCacheSize() const;
    BlockCache::ptr cache() const; // thread-safe
    void        discardOutside(Signal::Interval I);
    bool        failed_allocation();

    bool isVisible() const;
    void setVisible(bool v);

    BlockLayout block_layout() const;
    VisualizationParams::const_ptr visualization_params() const;

    void length(double length);
    void block_layout(BlockLayout block_layout);
    void visualization_params(VisualizationParams::const_ptr visualization_params);

private:
    BlockLayout block_layout_;
    VisualizationParams::const_ptr visualization_params_;

    bool failed_allocation_ = false;

    BlockCache::ptr cache_;
    std::set<pBlock> to_remove_;
    std::unique_ptr<BlockManagement::BlockFactory> block_factory_;
    std::unique_ptr<BlockManagement::BlockInitializer> block_initializer_;

    Signal::Intervals
        recently_created_,
        missing_data_, missing_data_next_; // "double buffered", waiting for glFlush

    bool
        _is_visible;

    unsigned
        _frame_counter;

    double
        _prev_length;

    Position
            _max_sample_size;

    /**
      Update 'b':S last_frame_used so that it wont yet be freed.
      */
    void        poke( pBlock b );


    void        removeBlock( pBlock b );
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
