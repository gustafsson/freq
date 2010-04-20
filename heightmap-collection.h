#ifndef HEIGHTMAPCOLLECTION_H
#define HEIGHTMAPCOLLECTION_H

#include "heightmap-reference.h"
#include "heightmap-vbo.h"
#include "signal-samplesintervaldescriptor.h"
#include "signal-source.h"
#include "signal-worker.h"
#include "tfr-chunk.h"

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


/*
#include <list>
#include <set>
#include <boost/shared_ptr.hpp>
#include <tvector.h>
#include <vector>
#include "transform.h"
#include "position.h"
#include "signal-worker.h"
  */
namespace Heightmap {

class Block {
public:
    Block( Heightmap::Reference ref ): ref(ref) {}

    float sample_rate();
    float nFrequencies();

    // Zoom level for this slot, determines size of elements
	Heightmap::Reference ref;
    unsigned frame_number_last_used;
    pGlBlock glblock;

    typedef boost::shared_ptr<GpuCpuData<float> > pData;
    pData prepared_data;

    Signal::SamplesIntervalDescriptor isd;
};
typedef boost::shared_ptr<Block> pBlock;


/**
  put is used to insert information into this collection.
  getBlock is used to extract blocks for rendering.
  */
class Collection: public Signal::WorkerCallback {
public:
    Collection();


    // WorkerCallback: Implementations of virtual methods

    /**
      Releases all GPU resources allocated by Heightmap::Collection.
      */
    virtual void reset() { gc(); }

    /**
      @see put( pBuffer, pSource );
      */
    virtual void put( Signal::pBuffer b) { put(b, Signal::pSource()); }

    /**
      Computes the Cwt and updates the cache of blocks.
      */
    virtual void put( Signal::pBuffer, Signal::pSource );


    /**
      scales_per_block and samples_per_block are constants deciding how many blocks
      are to be created.
      */
    unsigned    scales_per_block() { return _scales_per_block; }
    void        scales_per_block(unsigned v);
    unsigned    samples_per_block() { return _samples_per_block; }
    void        samples_per_block(unsigned v);

    /**
      If a fast source is provided collection will use this source to when allocating
      new new blocks. A Buffer over the entire block will be fetched and the initial
      value of the block will be computed by a fast windowed fourier transform.

      This source could for instance be the original source without any operations
      applied to it.

      TODO: walk through the worker source instead and look for a cache source and
          query its InvalidSampleDescriptor to make sure that the required samples
          are readily available. Otherwise, continue down and look for another cache
          or a source that is not an operation. I.e an original source such as
          MicrophoneRecroder or Audiofile, these sources are fast.
      */
    Signal::pSource fast_source() { return _fast_source; }
    void            fast_source( Signal::pSource v ) { _fast_source = v; }
    /**
      getBlock increases a counter for each block that hasn't been computed yet.
      */
    unsigned    read_unfinished_count();

    /**
      As the cwt is of finite length and finite sample rate there is a smallest
      and a largest sample size that is meaningful for the heightmap to render.
      */
    Position min_sample_size();
    Position max_sample_size();


    /**
      Returns a Reference for a block containing 'p' in which a block element
      is as big as possible yet smaller than or equal to 'sampleSize'.
      */
    Reference   findReference( Position p, Position sampleSize );

    /**
      Get a heightmap block. If the reference doesn't exist it is created

      This method is used by Heightmap::Renderer to get the heightmap data of
      blocks that has been decided for rendering.
      */
    pBlock      getBlock( Reference ref, bool* finished_block=0 );


    void        gc();

private:
    unsigned
        _samples_per_block,
        _scales_per_block,
        _unfinished_count,
        _frame_counter; // TODO shouldn't need _frame_counter
    Signal::pSource _fast_source;

    /**
      The cache contains as many blocks as there are space for in the GPU ram.
      If allocation of a new block fails to be allocated
            1) all unused blocks are freed.
            2) if no unused blocks are found and _cache is non-empty, the entire _cache is cleared.
            3) if _cache is empty, Sonic AWE is terminated with an OpenGL error.
      */
    std::vector<pBlock> _cache;

    /**
      Attempts to allocate a new block.
      */
    pBlock      attempt( Reference ref );

    /**
      Creates a new block.
      */
    pBlock      createBlock( Reference ref );
    void        prepareFillStft( pBlock block );

    /**
      Update the slope texture used by the vertex shader. Called when height data has been updated.
      */
    void        updateSlope( pBlock block, unsigned cuda_stream );

    /**
      Add block information from Cwt transform.
      */
    void        mergeBlock( pBlock outBlock, Tfr::pChunk inChunk, unsigned cuda_stream, bool save_in_prepared_data = false );

    /**
      Add block information from another block.
      */
    void        mergeBlock( pBlock outBlock, pBlock inBlock, unsigned cuda_stream );
};
typedef boost::shared_ptr<Collection> pCollection;

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
