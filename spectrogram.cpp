#include "spectrogram.h"
#include "spectrogram-vbo.h"  // breaks model-view-controller, but I want the rendering context associated with a spectrogram block to be removed when the cached block is removed

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


Spectrogram::Reference Spectrogram::findReferenceCanonical( Position p, Position sampleSize )
{
    // doesn't ASSERT(r.containsSpectrogram() && !r.toLarge())
    Spectrogram::Reference r(this);

    if (p.time < 0) p.time=0;
    if (p.scale < 0) p.scale=0;

    r.log2_samples_size = tvector<2,int>( floor(log2( sampleSize.time )), floor(log2( sampleSize.scale )) );
    r.chunk_index = tvector<2,unsigned>(p.time / _samples_per_block * pow(2, -r.log2_samples_size[0]),
                                        p.scale / _scales_per_block * pow(2, -r.log2_samples_size[1]));

    return r;
}

Spectrogram::Reference Spectrogram::findReference( Position p, Position sampleSize )
{
    Spectrogram::Reference r(this);

    // make sure the reference becomes valid
    pWaveform wf = transform()->original_waveform();
    float length = wf->length();

    // Validate requested sampleSize
    sampleSize.time = fabs(sampleSize.time);
    sampleSize.scale = fabs(sampleSize.scale);

    Position minSampleSize = min_sample_size();
    Position maxSampleSize = max_sample_size();
    if (sampleSize.time > maxSampleSize.time)
        sampleSize.time = maxSampleSize.time;
    if (sampleSize.scale > maxSampleSize.scale)
        sampleSize.scale = maxSampleSize.scale;
    if (sampleSize.time < minSampleSize.time)
        sampleSize.time = minSampleSize.time;
    if (sampleSize.scale < minSampleSize.scale)
        sampleSize.scale = minSampleSize.scale;

    // Validate requested poistion
    if (p.time < 0) p.time=0;
    if (p.time > length) p.time=length;
    if (p.scale < 0) p.scale=0;
    if (p.scale > 1) p.scale=1;

    // Compute sample size
    r.log2_samples_size = tvector<2,int>( floor(log2( sampleSize.time )), floor(log2( sampleSize.scale )) );
    r.chunk_index = tvector<2,unsigned>(0,0);

    // Validate sample size
    Position a,b; r.getArea(a,b);
    if (b.time < minSampleSize.time*_samples_per_block )                r.log2_samples_size[0]++;
    if (b.scale < minSampleSize.scale*_scales_per_block )               r.log2_samples_size[1]++;
    if (b.time >= maxSampleSize.time*_samples_per_block && 0<length )   r.log2_samples_size[0]--;
    if (b.scale >= maxSampleSize.scale*_scales_per_block )              r.log2_samples_size[1]--;

    // Compute chunk index
    r.chunk_index = tvector<2,unsigned>(p.time / _samples_per_block * pow(2, -r.log2_samples_size[0]),
                                        p.scale / _scales_per_block * pow(2, -r.log2_samples_size[1]));

    // Validate chunk index
    r.getArea(a,b);
    if (a.time >= length && 0<length)   r.chunk_index[0]--;
    if (a.scale == 1)                   r.chunk_index[1]--;

    // Test result
    // ASSERT(r.containsSpectrogram() && !r.toLarge());

    return r;
}

Spectrogram::Position Spectrogram::min_sample_size() {
    return Position( 1.f/transform()->original_waveform()->sample_rate(),
                        1.f/(transform()->number_of_octaves() * transform()->scales_per_octave()) );
}

Spectrogram::Position Spectrogram::max_sample_size() {
    pWaveform wf = transform()->original_waveform();
    float length = wf->length();

    return Position( 2.f*length/_samples_per_block,
                        2.f/_scales_per_block );
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
        attempt->vbo.reset( new SpectrogramVbo( this ));
        Slot s = { attempt, time(0) };
        _cache.push_back( s );
        block = attempt;
    }
    catch (const CudaException& x )
    {
        // Try to reuse an old block instead
        // Look for the oldest block
        time_t oldest = time(0);
        Slot* pb=0;
        BOOST_FOREACH( Slot& b, _cache ) {
            if ( 0< difftime(oldest, b.last_access)) {
                oldest = b.last_access;
                pb = &b;
            }
        }
        if (pb) {
            block = pb->block;
            pb->last_access = time(0);
        }
    }
    if ( 0 == block.get())
        return block; // return null-pointer

    // Reset block with dummy values
    SpectrogramVbo::pHeight h = block->vbo->height();
    float* p = h->data->getCpuMemory();
    for (unsigned s = 0; s<_samples_per_block; s++) {
        for (unsigned f = 0; f<_scales_per_block; f++) {
            p[ f*_samples_per_block + s] = sin(s*1./_samples_per_block)*cos(f*1./_scales_per_block);
        }
    }
    h->data->getCudaGlobal();

    // TODO Compute block
    return block;
}


bool Spectrogram::Reference::operator==(const Spectrogram::Reference &b) const
{
    return log2_samples_size == b.log2_samples_size
            && chunk_index == b.chunk_index
            && _spectrogram == b._spectrogram;
}

void Spectrogram::Reference::getArea( Position &a, Position &b) const
{
    Position blockSize( _spectrogram->samples_per_block() * pow(2,log2_samples_size[0]),
                        _spectrogram->scales_per_block() * pow(2,log2_samples_size[1]));
    a.time = blockSize.time * chunk_index[0];
    a.scale = blockSize.scale * chunk_index[1];
    b.time = a.time + blockSize.time;
    b.scale = a.scale + blockSize.scale;
}

/* child references */
Spectrogram::Reference Spectrogram::Reference::left() {
    Reference r = *this;
    r.log2_samples_size[0]--;
    r.chunk_index[0]<<=1;
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
    return r;
}
Spectrogram::Reference Spectrogram::Reference::sibbling2() {
    Reference r = *this;
    r.chunk_index[1]^=1;
    return r;
}
Spectrogram::Reference Spectrogram::Reference::sibbling3() {
    Reference r = *this;
    r.chunk_index[0]^=1;
    r.chunk_index[1]^=1;
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

Spectrogram::Reference::Reference(Spectrogram *spectrogram)
:   _spectrogram(spectrogram)
{}

bool Spectrogram::Reference::containsSpectrogram() const
{
    Position a, b;
    getArea( a, b );

    if (b.time-a.time < _spectrogram->min_sample_size().time*_spectrogram->_samples_per_block )
        return false;
    if (b.scale-a.scale < _spectrogram->min_sample_size().scale*_spectrogram->_scales_per_block )
        return false;

    pTransform t = _spectrogram->transform();
    pWaveform wf = t->original_waveform();
    if (a.time >= wf->length() )
        return false;

    if (a.scale >= 1)
        return false;

    return true;
}

bool Spectrogram::Reference::toLarge() const
{
    Position a, b;
    getArea( a, b );
    pTransform t = _spectrogram->transform();
    pWaveform wf = t->original_waveform();
    if (b.time > 2 * wf->length() && b.scale >= 5 )
        return true;
    return false;
}
