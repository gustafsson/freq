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
*/

typedef boost::shared_ptr<class Filter> FilterPtr;

class Spectogram_chunk: public Transform_chunk
{
public:
    Spectogram_chunk(float start_time, float end_time,
                  float lowest_scale, float highest_scale,
                  int fzoom, int tzoom );

    // Zoom level for this slot, determines size of elements
    int fzoom, tzoom;

    // Contains start float start_t, start_f;
    float _start_time, _lowest_scale;
    float _end_time, _highest_scale;
};

struct SpectogramSlot {
    SpectogramBlock block;
    unsigned last_access;
};

class Spectogram
{
public:
    Spectogram( boost::shared_ptr<Waveform> waveform, unsigned waveform_channel=0 );

    void garbageCollect();
    SpectogramBlock* getBlock( float t, float f, float dt, float df);

    FilterPtr filter_chain;

private:
    // Spectogram_chunk computeStft( pWaveform waveform, float );
    boost::shared_ptr<Waveform> _original_waveform;
    unsigned _original_waveform_channel;

    // Slots with SpectogramBlock:s, as many as there are space for in the GPU ram
    list<SpectogramSlot> _slots;
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
