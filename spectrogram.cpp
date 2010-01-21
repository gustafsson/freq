#include "spectrogram.h"

#include <boost/foreach.hpp>
#include <CudaException.h>

using namespace std;


Spectrogram::Spectrogram( pTransform transform, unsigned samples_per_block, unsigned scales_per_block )
:   _transform(transform),
    _samples_per_block(samples_per_block),
    _scales_per_block(scales_per_block)
{}


void Spectrogram::scales_per_block(unsigned v) {
    _cache.clear();
    _scales_per_block=v;
}

void Spectrogram::samples_per_block(unsigned v) {
    _cache.clear();
    _samples_per_block=v;
}


Spectrogram::Reference Spectrogram::findReference( Position p, Position sampleSize )
{
    Spectrogram::Reference r;

    r.log2_samples_size = tvector<2,int>( floor(log2( sampleSize.time )), floor(log2( sampleSize.scale )) );
    r.chunk_index = tvector<2,unsigned>(p.time / _samples_per_block * pow(2, -r.log2_samples_size[0]),
                                        p.scale / _scales_per_block * pow(2, -r.log2_samples_size[1]));

    return r;
}


Spectrogram::pBlock Spectrogram::getBlock( Spectrogram::Reference ref) {
    // Look among cached blocks for this reference
    BOOST_FOREACH( Slot& b, _cache ) {
        if (b.block->ref == ref) {
            b.last_access = time(0);
            return b.block;
        }
    }

    // Try to allocate a new block
    Spectrogram::pBlock block;
    try {
        Spectrogram::pBlock attempt( new Spectrogram::Block(ref));
        attempt->transform_data.reset( new GpuCpuData<float>(0, make_cudaExtent(_samples_per_block, _scales_per_block, 1)));
        block = attempt;
    }
    catch (const CudaException& x )
    {
        // Try to reuse an old block instead
        // Look for the oldest block
        time_t oldest = time(0);
        BOOST_FOREACH( Slot& b, _cache ) {
            if ( 0< difftime(oldest, b.last_access)) {
                oldest = b.last_access;
                block = b.block;
            }
        }
    }
    if ( 0 == block.get())
        return block; // return null-pointer

    // TODO Compute block
    return block;
}


bool Spectrogram::Reference::operator==(const Spectrogram::Reference &b) const
{
    return log2_samples_size == b.log2_samples_size
            && chunk_index == b.chunk_index
            && _parent == b._parent;
}

void Spectrogram::Reference::getArea( Position &a, Position &b) const
{
    Position blockSize( _parent->samples_per_block() * pow(2,log2_samples_size[0]),
                        _parent->scales_per_block() * pow(2,log2_samples_size[1]));
    a.time = blockSize.time*chunk_index[0];
    a.scale = blockSize.scale*chunk_index[1];
    b.time = a.time + blockSize.time;
    b.scale = a.scale + blockSize.scale;
}

bool Spectrogram::Reference::valid() const
{
    return 0!=_parent.get();
}

/* child references */
Spectrogram::Reference Spectrogram::Reference::left() {
    Reference r = *this;
    r.log2_samples_size[0]--;
    (r.chunk_index[0]<<=1)++;
    return r;
}
Spectrogram::Reference Spectrogram::Reference::right() {
    Reference r = *this;
    r.log2_samples_size[0]--;
    (r.chunk_index[0]<<=1)++;
    return r;
}
Spectrogram::Reference Spectrogram::Reference::top() {
    Reference r = *this;
    r.log2_samples_size[1]--;
    r.chunk_index[1]<<=1;
    return r;
}
Spectrogram::Reference Spectrogram::Reference::bottom() {
    Reference r = *this;
    r.log2_samples_size[1]--;
    (r.chunk_index[1]<<=1)++;
    return r;
}

/* sibblings, 3 other references who share the same parent */
Spectrogram::Reference Spectrogram::Reference::sibbling1() {
    Reference r = *this;
    r.chunk_index[0]^=1;
    r.validate();
    return r;
}
Spectrogram::Reference Spectrogram::Reference::sibbling2() {
    Reference r = *this;
    r.chunk_index[1]^=1;
    r.validate();
    return r;
}
Spectrogram::Reference Spectrogram::Reference::sibbling3() {
    Reference r = *this;
    r.chunk_index[0]^=1;
    r.chunk_index[1]^=1;
    r.validate();
    return r;
}

/* parent */
Spectrogram::Reference Spectrogram::Reference::parent() {
    Reference r = *this;
    r.log2_samples_size[0]++;
    r.log2_samples_size[1]++;
    r.chunk_index[0]>>=1;
    r.chunk_index[1]>>=1;
    return r;
}

Spectrogram::Reference::Reference() {}

void Spectrogram::Reference::validate()
{
    Position a, b;
    getArea( a, b );
    pTransform t = _parent->transform();
    pWaveform wf = t->original_waveform();
    float length = wf->number_of_samples() / (float)wf->sample_rate();
    if (a.time > length)
        _parent.reset(); // invalidate by setting the reference to a null pointer

    float height = t->scales_per_octave() * t->number_of_octaves();
    if (a.scale > height)
        _parent.reset();
}
