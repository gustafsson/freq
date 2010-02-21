#include "spectrogram.h"
#include "spectrogram-vbo.h"  // breaks model-view-controller, but I want the rendering context associated with a spectrogram block to be removed when the cached block is removed
#include "spectrogram-slope.cu.h"
#include "spectrogram-block.cu.h"

#include <boost/foreach.hpp>
#include <CudaException.h>
#include <GlException.h>
#include <math.h>
#include <msc_stdc.h>

using namespace std;


Spectrogram::Spectrogram( pTransform transform, unsigned samples_per_block, unsigned scales_per_block )
:   _transform(transform),
    _samples_per_block(samples_per_block),
    _scales_per_block(scales_per_block),
    _frame_counter(0)
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
    //printf("%d %d\n", r.log2_samples_size[0], r.log2_samples_size[1]);

    // Validate sample size
    Position a,b; r.getArea(a,b);
    if (b.time < minSampleSize.time*_samples_per_block )                r.log2_samples_size[0]++;
    if (b.scale < minSampleSize.scale*_scales_per_block )               r.log2_samples_size[1]++;
    if (b.time > maxSampleSize.time*_samples_per_block && 0<length )    r.log2_samples_size[0]--;
    if (b.scale > maxSampleSize.scale*_scales_per_block )               r.log2_samples_size[1]--;
    //printf("%d %d\n", r.log2_samples_size[0], r.log2_samples_size[1]);

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
                        max(minima.scale, 1.f/_scales_per_block) );
}

Spectrogram::pBlock Spectrogram::getBlock( Spectrogram::Reference ref) {
    // Look among cached blocks for this reference
    pBlock block;
    BOOST_FOREACH( pBlock& b, _cache ) {
        if (b->ref == ref) {
            block = b;
            b->frame_number_last_used = _frame_counter;

            break;
        }
    }

    pBlock result = block;
    try {
        bool need_slope = false;
        if (0 == block.get()) {
            block = createBlock( ref );
            need_slope = true;
        }

        if (0 != block.get()) {
            if ( 0 /* block for complete block */ ) {
                if (getNextInvalidChunk(block,0))
                    computeBlock(block);
            } else if (1 /* partial blocks, single threaded */ ) {
                unsigned previous_chunk_index = (unsigned)-1;
                pTransform_chunk chunk = _transform->previous_chunk(previous_chunk_index);

                if (chunk && isInvalidChunk( block, previous_chunk_index)) {
                    mergeBlock( block, chunk, 0 );
                    block->valid_chunks.insert( previous_chunk_index );
                    need_slope = true;
                } else if (getNextInvalidChunk(block,0)) {
                    if (0==_unfinished_count) {
                        computeBlockOneChunk( block, 0 );

                        need_slope = true;
                    }
                    _unfinished_count++;
                }
#ifdef MULTITHREADED_SONICAWE
            } else if (0 /* partial block, multithreaded, multiple GPUs*/ ) {
                bool enqueue = false;
                {
                    Block::pData prepared_data = block->prepared_data;
                    if (0 != prepared_data.get()) {
                        SpectrogramVbo::pHeight h = block->vbo->height();

                        BOOST_ASSERT( h->data->getSizeInBytes1D() == prepared_data->getSizeInBytes1D() );
                        memcpy(h->data->getCpuMemory(),
                               prepared_data->getCpuMemory(),
                               h->data->getSizeInBytes1D());

                        enqueue = true;
                        need_slope = true;
                    }
                }
                if (getNextInvalidChunk( block, 0 )) {
                    enqueue = true;
                    _unfinished_count++;
                }
                if (enqueue)
                    block_worker()->filo_enqueue( block );
#endif
            }

            if (need_slope) {
                computeSlope( block, 0 );
            }
        }

        result = block;
    } catch (const CudaException &) {
    } catch (const GlException &) {
    }

    return result;
}

#ifdef MULTITHREADED_SONICAWE
Spectrogram::BlockWorker* Spectrogram::block_worker() {
    if ( 0 == _block_worker ) {
        _block_worker.reset(new BlockWorker( this ));
    }

    return _block_worker.get();
}

Spectrogram::BlockWorker::BlockWorker( Spectrogram* parent, unsigned cuda_stream )
:   _cuda_stream( cuda_stream ),
    _spectrogram( parent )
{
    start();
}

void Spectrogram::BlockWorker::filo_enqueue( Spectrogram::pBlock block )
{
    {
        boost::lock_guard<boost::mutex> lock(_mut);
        _next_block = block;
    }

    _cond.notify_one();
}

Spectrogram::pBlock Spectrogram::BlockWorker::wait_for_data_to_process()
{
    boost::unique_lock<boost::mutex> lock(_mut);
    pBlock data;
    while(0==data.get() || !_spectrogram->getNextInvalidChunk(data, 0))
    {
        _cond.wait(lock);
        data = _next_block;

        if (0!=data.get() && !_spectrogram->getNextInvalidChunk(data, 0))
            data->prepared_data.reset();
    }
    return data;
}

void Spectrogram::BlockWorker::process_data(pBlock block)
{
    _spectrogram->computeBlockOneChunk( block, _cuda_stream, true );
}

void Spectrogram::BlockWorker::run()
{
    while (true) {
        process_data( wait_for_data_to_process() );
    }
}
#endif
Spectrogram::pBlock Spectrogram::createBlock( Spectrogram::Reference ref )
{
    // Try to allocate a new block
    pBlock block;
    try {
        pBlock attempt( new Spectrogram::Block(ref));
        attempt->vbo.reset( new SpectrogramVbo( this ));
        SpectrogramVbo::pHeight h = attempt->vbo->height();
        SpectrogramVbo::pSlope sl = attempt->vbo->slope();

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        block = attempt;
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

    pBlock result;
    try {
        // set to zero
        SpectrogramVbo::pHeight h = block->vbo->height();
        cudaMemset( h->data->getCudaGlobal().ptr(), 0, h->data->getSizeInBytes1D() );

        if ( 1 /* create from others */ ) {
            // TODO not the best idea to use all caches, but it's better than to use no caches
            BOOST_FOREACH( pBlock& b, _cache ) {
                mergeBlock( block, b, 0 );
            }

        } else if ( 0 /* set dummy values */ ) {
            SpectrogramVbo::pHeight h = block->vbo->height();
            float* p = h->data->getCpuMemory();
            for (unsigned s = 0; s<_samples_per_block; s++) {
                for (unsigned f = 0; f<_scales_per_block; f++) {
                    p[ f*_samples_per_block + s] = sin(s*10./_samples_per_block)*cos(f*10./_scales_per_block);
                }
            }
        }

        result = block;
    }
    catch (const CudaException& x )
    { }
    catch (const GlException& x )
    { }

    if ( 0 == result.get())
        return result; // return null-pointer

    _cache.push_back( result );

    return result;
}

void Spectrogram::computeBlock( Spectrogram::pBlock block ) {
    while( computeBlockOneChunk( block, 0 ) );
    computeSlope(block, 0);
}

bool Spectrogram::computeBlockOneChunk( Spectrogram::pBlock block, unsigned cuda_stream, bool prepare ) {
    Transform::ChunkIndex n;
    if (getNextInvalidChunk( block, &n )) {
        pTransform_chunk chunk = _transform->getChunk(n, cuda_stream);
        mergeBlock( block, chunk, cuda_stream, prepare );
        block->valid_chunks.insert( n );

        return true;
    }

    return false;
}

void Spectrogram::computeSlope( Spectrogram::pBlock block, unsigned cuda_stream ) {
    SpectrogramVbo::pHeight h = block->vbo->height();
    cudaCalculateSlopeKernel( h->data->getCudaGlobal().ptr(), block->vbo->slope()->data->getCudaGlobal().ptr(), _samples_per_block, _scales_per_block, cuda_stream );
}

bool Spectrogram::getNextInvalidChunk( pBlock block, Transform::ChunkIndex* on )
{
    Position a, b;
    block->ref.getArea( a, b );

    unsigned
        start = a.time * _transform->original_waveform()->sample_rate(),
        end = b.time * _transform->original_waveform()->sample_rate();

    for (Transform::ChunkIndex n = _transform->getChunkIndex(start);
         n*_transform->samples_per_chunk() < end;
         n++)
    {
        if (block->valid_chunks.find( n ) == block->valid_chunks.end()) {
            if(on) *on = n;
            return true;
        }
    }
    return false;
}

bool Spectrogram::isInvalidChunk( pBlock block, Transform::ChunkIndex n )
{
    Position a, b;
    block->ref.getArea( a, b );

    unsigned
        start = a.time * _transform->original_waveform()->sample_rate(),
        end = b.time * _transform->original_waveform()->sample_rate();

    bool r = block->valid_chunks.find( n ) == block->valid_chunks.end();
    r &=_transform->getChunkIndex(start)<=n && n<=_transform->getChunkIndex(end-1);
    return r;
}

void Spectrogram::invalidate_range(float start_time, float end_time)
{
    unsigned start = start_time*_transform->original_waveform()->sample_rate();
    unsigned end = end_time*_transform->original_waveform()->sample_rate();
    BOOST_FOREACH( pBlock& b, _cache ) {
        for (Transform::ChunkIndex n = _transform->getChunkIndex(start);
             n*_transform->samples_per_chunk() < end;
             n++)
        {
            b->valid_chunks.erase( n );
        }
    }
}

void Spectrogram::mergeBlock( Spectrogram::pBlock outBlock, pTransform_chunk inChunk, unsigned cuda_stream, bool save_in_prepared_data) {
    boost::shared_ptr<GpuCpuData<float> > outData;

    if (!save_in_prepared_data)
        outData = outBlock->vbo->height()->data;
    else {
        if (outBlock->prepared_data)
            outData = outBlock->prepared_data;
        else
            outData.reset( new GpuCpuData<float>(0, make_cudaExtent( _samples_per_block, _scales_per_block, 1 ), GpuCpuVoidData::CudaGlobal ) );
    }

    Position a, b;
    outBlock->ref.getArea( a, b );

    float in_sample_rate = inChunk->sample_rate;
    float out_sample_rate = outBlock->sample_rate();
    float in_frequency_resolution = inChunk->nScales();
    float out_frequency_resolution = outBlock->nFrequencies();
    float in_offset = max(0.f, (a.time-inChunk->startTime()))*in_sample_rate;
    float out_offset = max(0.f, (inChunk->startTime()-a.time))*out_sample_rate;

    in_offset += inChunk->first_valid_sample;
    blockMergeChunk( inChunk->transform_data->getCudaGlobal(),
                      outData->getCudaGlobal(false),
                           in_sample_rate,
                           out_sample_rate,
                           in_frequency_resolution,
                           out_frequency_resolution,
                           in_offset,
                           out_offset,
                           inChunk->n_valid_samples,
                           cuda_stream);

    if (save_in_prepared_data) {
        outData->getCpuMemory();
        outBlock->prepared_data = outData;
    }
}

void Spectrogram::mergeBlock( Spectrogram::pBlock outBlock, Spectrogram::pBlock inBlock, unsigned cuda_stream )
{
    SpectrogramVbo::pHeight out_h = outBlock->vbo->height();
    SpectrogramVbo::pHeight in_h = inBlock->vbo->height();

    Position in_a, in_b, out_a, out_b;
    inBlock->ref.getArea( in_a, in_b );
    outBlock->ref.getArea( out_a, out_b );

    if (in_a.time >= out_b.time)
        return;
    if (in_b.time <= out_a.time)
        return;
    if (in_a.scale >= out_b.scale)
        return;
    if (in_b.scale <= out_a.scale)
        return;

    float in_sample_rate = inBlock->sample_rate();
    float out_sample_rate = outBlock->sample_rate();
    float in_frequency_resolution = inBlock->nFrequencies();
    float out_frequency_resolution = outBlock->nFrequencies();
    float in_offset = max(0.f, (out_a.time-in_a.time))*in_sample_rate;
    float out_offset = max(0.f, (in_a.time-out_a.time))*out_sample_rate;

    blockMerge( in_h->data->getCudaGlobal(),
                out_h->data->getCudaGlobal(),

                           in_sample_rate,
                           out_sample_rate,
                           in_frequency_resolution,
                           out_frequency_resolution,
                           in_offset,
                           out_offset,
                           cuda_stream);
}

float Spectrogram::Block::sample_rate() {
    Position a, b;
    ref.getArea( a, b );
    return pow(2, -ref.log2_samples_size[0]) + 1/(b.time-a.time);
}

float Spectrogram::Block::nFrequencies() {
    return pow(2, -ref.log2_samples_size[1]);
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
    //float msss = _spectrogram->min_sample_size().scale;
    //unsigned spb = _spectrogram->_scales_per_block;
    //float ms = msss*spb;
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
