#ifndef SPECTROGRAM_H
#define SPECTROGRAM_H

/*
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

#include <list>
#include <boost/shared_ptr.hpp>
#include <tvector.h>
#include <vector>
#include "transform.h"
#include "waveform.h"

typedef boost::shared_ptr<class Filter> pFilter;
typedef boost::shared_ptr<class Spectrogram> pSpectrogram;
typedef boost::shared_ptr<class SpectrogramVbo> pSpectrogramVbo;

struct position {
    typedef tvector<2,int> Log2samplesPerUnit;
    typedef tvector<2,unsigned> ChunkPosition;
    Log2samplesPerUnit zoom;
    ChunkPosition pos;
};

class Spectrogram
{
public:
    class Position;
    class Reference;
    class Block;

    typedef boost::shared_ptr<Block> pBlock;

    Spectrogram( pTransform transform, unsigned samples_per_block, unsigned scales_per_block );

    void garbageCollect();

    pBlock      getBlock( Reference ref );
    pTransform  transform() const { return _transform; }
    void        gc();

    /**
      Returns a Reference for a block containing 'p' in which a block element
      is as big as possible yet smaller than or equal to 'sampleSize'.
      */
    Reference   findReference( Position p, Position sampleSize );

    pFilter filter_chain;

    unsigned scales_per_block() { return _scales_per_block; }
    unsigned samples_per_block() { return _samples_per_block; }
    void scales_per_block(unsigned v);
    void samples_per_block(unsigned v);

    Position min_sample_size();
    Position max_sample_size();
private:
    // Spectrogram_chunk computeStft( pWaveform waveform, float );
    pTransform _transform;
    unsigned _samples_per_block;
    unsigned _scales_per_block;

    // Slots with Spectrogram::Block:s, as many as there are space for in the GPU ram
    std::vector<pBlock> _cache;

    Reference   findReferenceCanonical( Position p, Position sampleSize );
    void        computeBlock( Spectrogram::pBlock block );
    void        mergeBlock( Spectrogram::pBlock outBlock, pTransform_chunk inChunk );
};

class Spectrogram::Position {
public:
    float time, scale;

    Position() { }
    Position(float time, float scale):time(time), scale(scale) {}

    tvector<2, float> operator()() { return tvector<2, float>(time, scale); }
};

class Spectrogram::Reference {
public:
    tvector<2,int> log2_samples_size;
    tvector<2,unsigned> chunk_index;

    bool operator==(const Reference &b) const;
    void getArea( Position &a, Position &b) const;
    unsigned sampleOffset() const;
    unsigned scaleOffset() const;

    bool containsSpectrogram() const;
    bool toLarge() const;

    /* child references */
    Reference left();
    Reference right();
    Reference top();
    Reference bottom();

    /* sibblings, 3 other references who share the same parent */
    Reference sibbling1();
    Reference sibbling2();
    Reference sibbling3();

    /* parent */
    Reference parent();
private:
    Reference( Spectrogram* parent );

    friend class Spectrogram;

    Spectrogram* _spectrogram;
};

class Spectrogram::Block
{
public:
    Block( Spectrogram::Reference ref ): ref(ref) {}

    // Zoom level for this slot, determines size of elements
    Spectrogram::Reference ref;
    unsigned frame_number_last_used;

    pSpectrogramVbo vbo;
};

#endif // SPECTROGRAM_H
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
