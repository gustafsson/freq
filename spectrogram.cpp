#include "spectrogram.h"
#include "spectrogram-vbo.h"  // breaks model-view-controller, but I want the rendering context associated with a spectrogram block to be removed when the cached block is removed
#include "spectrogram-slope.cu.h"
#include "spectrogram-block.cu.h"

#include <boost/foreach.hpp>
#include <CudaException.h>
#include <GlException.h>

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
    printf("%d %d\n", r.log2_samples_size[0], r.log2_samples_size[1]);

    // Validate sample size
    Position a,b; r.getArea(a,b);
    if (b.time < minSampleSize.time*_samples_per_block )                r.log2_samples_size[0]++;
    if (b.scale < minSampleSize.scale*_scales_per_block )               r.log2_samples_size[1]++;
    if (b.time > maxSampleSize.time*_samples_per_block && 0<length )    r.log2_samples_size[0]--;
    if (b.scale > maxSampleSize.scale*_scales_per_block )               r.log2_samples_size[1]--;
    printf("%d %d\n", r.log2_samples_size[0], r.log2_samples_size[1]);

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
    Position minima=min_sample_size();

    return Position( max(minima.time, 2.f*length/_samples_per_block),
                        max(minima.scale, 2.f/_scales_per_block) );
}


Spectrogram::pBlock Spectrogram::getBlock( Spectrogram::Reference ref) {
    // Look among cached blocks for this reference
    BOOST_FOREACH( pBlock& b, _cache ) {
        if (b->ref == ref) {
            return b;
        }
    }

    // Try to allocate a new block
    Spectrogram::pBlock block;
    try {
        Spectrogram::pBlock attempt( new Spectrogram::Block(ref));
        attempt->vbo.reset( new SpectrogramVbo( this ));
        SpectrogramVbo::pHeight h = attempt->vbo->height();
        SpectrogramVbo::pSlope sl = attempt->vbo->slope();

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        block = attempt;
        _cache.push_back( block );
    }
    catch (const CudaException& x )
    { }
    catch (const GlException& x )
    { }

    if ( 0 == block.get() && !_cache.empty()) {
        // Try to reuse an old block instead
        // Look for the oldest block

        unsigned oldest = _cache[0]->frame_number_last_used;

        BOOST_FOREACH( pBlock& b, _cache ) {
            if ( oldest > b->frame_number_last_used ) {
                oldest = b->frame_number_last_used;
                block = b;
            }
        }
    }

    if ( 0 == block.get())
        return block; // return null-pointer

    // Reset block with dummy values
    {
        SpectrogramVbo::pHeight h = block->vbo->height();
        float* p = h->data->getCpuMemory();
        for (unsigned s = 0; s<_samples_per_block; s++) {
            for (unsigned f = 0; f<_scales_per_block; f++) {
                p[ f*_samples_per_block + s] = sin(s*10./_samples_per_block)*cos(f*10./_scales_per_block);
            }
        }
        // Make sure it is moved to the gpu
        h->data->getCudaGlobal();

        // TODO Compute block
        // computeBlock( block );

        // Compute slope
        // TODO select cuda stream
        cudaCalculateSlopeKernel( h->data->getCudaGlobal().ptr(), block->vbo->slope()->data->getCudaGlobal().ptr(), _samples_per_block, _scales_per_block );
    }

    return block;
}

void Spectrogram::computeBlock( Spectrogram::pBlock block ) {
    Position a, b;
    block->ref.getArea( a, b );
    unsigned
        start = a.time * _transform->original_waveform()->sample_rate(),
        end = b.time * _transform->original_waveform()->sample_rate();

    SpectrogramVbo::pHeight h = block->vbo->height();
    for (unsigned t = start; t<end;) {
        Transform::ChunkIndex n = _transform->getChunkIndex(t);
        pTransform_chunk chunk = _transform->getChunk(n);
        mergeBlock( block, chunk );
        t += chunk->nSamples();
    }
}

void Spectrogram::mergeBlock( Spectrogram::pBlock outBlock, pTransform_chunk inChunk ) {
    SpectrogramVbo::pHeight h = outBlock->vbo->height();
    h->data->getCudaGlobal();
    Position a, b;
    outBlock->ref.getArea( a, b );

    float in_sample_rate = inChunk->sample_rate;
    float out_sample_rate = pow(2,-outBlock->ref.log2_samples_size[0]);
    float in_frequency_resolution = inChunk->nFrequencies();
    float out_frequency_resolution = pow(2,-outBlock->ref.log2_samples_size[0]);
    float in_offset = max(0.f,(a.time-inChunk->startTime()))*in_sample_rate;
    float out_offset = max(0.f,(inChunk->startTime()-a.time))*out_sample_rate;


    blockMerge( h->data->getCudaGlobal(),
                           inChunk->transform_data->getCudaGlobal(),
                           in_sample_rate,
                           out_sample_rate,
                           in_frequency_resolution,
                           out_frequency_resolution,
                           in_offset,
                           out_offset);
}

void Spectrogram::gc() {
    if (_cache.empty())
        return;

    unsigned latestFrame = _cache[0]->frame_number_last_used;
    BOOST_FOREACH( pBlock& b, _cache ) {
        if (latestFrame < b->frame_number_last_used)
            latestFrame = b->frame_number_last_used;
    }

    for (std::vector<pBlock>::iterator itr = _cache.begin(); itr!=_cache.end(); itr++)
    {
        if ((*itr)->frame_number_last_used < latestFrame) {
            itr = _cache.erase(itr);
        }
    }
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
    float msss = _spectrogram->min_sample_size().scale;
    unsigned spb = _spectrogram->_scales_per_block;
    float ms = msss*spb;
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
    if (b.time > 2 * wf->length() && b.scale > 2 )
        return true;
    return false;
}
