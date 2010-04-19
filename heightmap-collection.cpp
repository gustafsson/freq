#include "heightmap-collection.h"
/*
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


  */
namespace Heightmap {

///// HEIGHTMAP::BLOCK
float Block::
sample_rate()
{
    Position a, b;
    ref.getArea( a, b );
    return pow(2, -ref.log2_samples_size[0]) - 1/(b.time-a.time);
}

float Block::
nFrequencies()
{
    return pow(2, -ref.log2_samples_size[1]);
}


///// HEIGHTMAP::COLLECTION

Collection::
Collection( unsigned samples_per_block, unsigned scales_per_block )
:   _samples_per_block( samples_per_block ),
    _scales_per_block( scales_per_block )
{

}

void Collection::
put( pBuffer b, pSource s)
{
    try {
        // Get a chunk for this block
        Tfr::pChunk chunk;

        // If buffer comes directly from a Signal::FilterOperation
        Signal::FilterOperation* filterOp = dynamic_cast<Signal::FilterOperation*>(s.get());
        if (filterOp) {
            // use the Cwt chunk still stored in FilterOperation
            chunk = filterOp->previous_chunk();
        } else {
            // otherwise compute the Cwt of this block
            chunk = CwtSingleton::operator()( b );
        }

        // Update all blocks with this new chunk
        BOOST_FOREACH( pBlock& b, _cache ) {
            Position a,b;
            b->ref.getArea(a,b);
            if (b.time > b->start() && a.time < b->start()+b->length())
            {
                mergeBlock(b, chunk );
                computeSlope( b, 0 );
            }
        }
    } catch (const CudaException &) {
        // silently catch but don't bother to do anything
    } catch (const GlException &) {
        // silently catch but don't bother to do anything
    }
}


void Collection::
scales_per_block(unsigned v)
{
    _cache.clear();
    _scales_per_block=v;
}

void Collection::
samples_per_block(unsigned v)
{
    _cache.clear();
    _samples_per_block=v;
}

unsigned Collection::
read_unfinished_count()
{
    unsigned t = _unfinished_count;
    _unfinished_count = 0;
    _frame_counter++;

    BOOST_FOREACH( pBlock& b, _cache ) {
        b->vbo->unmap();
    }

    return t;
}

Position Collection::
min_sample_size()
{
    return Position( 1.f/transform()->original_waveform()->sample_rate(),
                        1.f/(transform()->number_of_octaves() * transform()->scales_per_octave()) );
}

Position Collection::
max_sample_size()
{
    Signal::pSource wf = transform()->original_waveform();
    float length = wf->length();
    Position minima=min_sample_size();

    return Position( max(minima.time, 2.f*length/_samples_per_block),
                        max(minima.scale, 1.f/_scales_per_block) );
}

static Reference findReferenceCanonical( Position p, Position sampleSize )
{
    // doesn't ASSERT(r.containsSpectrogram() && !r.toLarge())
    Reference r(this);

    if (p.time < 0) p.time=0;
    if (p.scale < 0) p.scale=0;

    r.log2_samples_size = tvector<2,int>( floor(log2( sampleSize.time )), floor(log2( sampleSize.scale )) );
    r.block_index = tvector<2,unsigned>(p.time / _samples_per_block * pow(2, -r.log2_samples_size[0]),
                                        p.scale / _scales_per_block * pow(2, -r.log2_samples_size[1]));

    return r;
}

Reference Collection::
findReference( Position p, Position sampleSize )
{
    Reference r(this);

    // make sure the reference becomes valid
    Signal::pSource wf = transform()->original_waveform();
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
    r.block_index = tvector<2,unsigned>(0,0);
    //printf("%d %d\n", r.log2_samples_size[0], r.log2_samples_size[1]);

    // Validate sample size
    Position a,b; r.getArea(a,b);
    if (b.time < minSampleSize.time*_samples_per_block )                r.log2_samples_size[0]++;
    if (b.scale < minSampleSize.scale*_scales_per_block )               r.log2_samples_size[1]++;
    if (b.time > maxSampleSize.time*_samples_per_block && 0<length )    r.log2_samples_size[0]--;
    if (b.scale > maxSampleSize.scale*_scales_per_block )               r.log2_samples_size[1]--;
    //printf("%d %d\n", r.log2_samples_size[0], r.log2_samples_size[1]);

    // Compute chunk index
    r.block_index = tvector<2,unsigned>(p.time / _samples_per_block * pow(2, -r.log2_samples_size[0]),
                                        p.scale / _scales_per_block * pow(2, -r.log2_samples_size[1]));

    // Validate chunk index
    r.getArea(a,b);
    if (a.time >= length && 0<length)   r.block_index[0]--;
    if (a.scale == 1)                   r.block_index[1]--;

    // Test result
    // ASSERT(r.containsSpectrogram() && !r.toLarge());

    return r;
}

pBlock Collection::
getBlock( Spectrogram::Reference ref, bool *finished_block)
{
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
        if (0 == block.get()) {
            block = createBlock( ref );
            computeSlope( block, 0 );
            _unfinished_count++;
        }

        if (0 != block.get()) {
            if (finished_block)
                *finished_block = !getNextInvalidChunk(block,0);
        }

        result = block;
    } catch (const CudaException &) {
    } catch (const GlException &) {
    }

    return result;
}

void Collection::
gc()
{
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

////// private


pBlock Collection::
attempt( Spectrogram::Reference ref )
{
    try {
        Spectrogram::pBlock attempt( new Spectrogram::Block(ref));
        attempt->vbo.reset( new SpectrogramVbo( this ));
        SpectrogramVbo::pHeight h = attempt->vbo->height();
        SpectrogramVbo::pSlope sl = attempt->vbo->slope();
        attempt->vbo->unmap();

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        return attempt;
    }
    catch (const CudaException& )
    { }
    catch (const GlException& )
    { }
    return Spectrogram::pBlock();
}


pBlock Collection::
createBlock( Spectrogram::Reference ref )
{
    Position a,b;
    ref.getArea(a,b);
    TaskTimer tt("Creating a new block [%g, %g]",a.time,b.time);
    // Try to allocate a new block
    pBlock block = attempt( this, ref );

    if ( 0 == block.get() && !_cache.empty()) {
        tt.info("Memory allocation failed, overwriting some older block");
        gc();
        block = attempt( this, ref );
    }

    if ( 0 == block.get()) {
        tt.info("Failed");
        return block; // return null-pointer
    }

    pBlock result;
    try {
        // set to zero
        SpectrogramVbo::pHeight h = block->vbo->height();
        cudaMemset( h->data->getCudaGlobal().ptr(), 0, h->data->getSizeInBytes1D() );

        if ( 1 /* create from others */ ) {
            TaskTimer tt(TaskTimer::LogVerbose, "Stubbing new block");

            // fill block by STFT
            {
                TaskTimer tt(TaskTimer::LogVerbose, "stft");
                fillStft( block );
            }

            if (0) {
                TaskTimer tt(TaskTimer::LogVerbose, "Preventing wavelet transform");
                Position a,b;
                block->ref.getArea(a,b);
                unsigned start = a.time * _transform->original_waveform()->sample_rate();
                unsigned end = b.time * _transform->original_waveform()->sample_rate();

                for (Transform::ChunkIndex n = _transform->getChunkIndex(start);
                     n <= _transform->getChunkIndex(end);
                     n++)
                {
                    block->valid_chunks.insert( n );
                }

            }

            if (0) {
                TaskTimer tt(TaskTimer::LogVerbose, "Fetching details");
                // start with the blocks that are just slightly more detailed
                BOOST_FOREACH( pBlock& b, _cache ) {
                    if (block->ref.log2_samples_size[0] == b->ref.log2_samples_size[0]+1 ||
                        block->ref.log2_samples_size[1] == b->ref.log2_samples_size[1]+1)
                    {
                        mergeBlock( block, b, 0 );
                    }
                }
            }

            if (0) {
                TaskTimer tt(TaskTimer::LogVerbose, "Fetching more details");
                // then try using the blocks that are even more detailed
                BOOST_FOREACH( pBlock& b, _cache ) {
                    if (block->ref.log2_samples_size[0] > b->ref.log2_samples_size[0] +1 ||
                        block->ref.log2_samples_size[1] > b->ref.log2_samples_size[1] +1)
                    {
                        mergeBlock( block, b, 0 );
                    }
                }
            }

            if (0) {
                TaskTimer tt(TaskTimer::LogVerbose, "Fetching low resolution");
                // then try to upscale other blocks
                BOOST_FOREACH( pBlock& b, _cache ) {
                    if (block->ref.log2_samples_size[0] < b->ref.log2_samples_size[0] ||
                        block->ref.log2_samples_size[1] < b->ref.log2_samples_size[1] )
                    {
                        mergeBlock( block, b, 0 );
                    }
                }
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

void Collection::
prepareFillStft( pBlock block ) {
    Position a, b;
    block->ref.getArea(a,b);
    float tmin = _transform->min_hz(),
          tmax = _transform->max_hz();

    unsigned in_stft_size;
    Signal::pBuffer stft = _transform->stft( a.time, b.time, &in_stft_size );

    float out_min_hz = exp(log(tmin) + (a.scale*(log(tmax)-log(tmin)))),
          out_max_hz = exp(log(tmin) + (b.scale*(log(tmax)-log(tmin)))),
          in_max_hz = _transform->original_waveform()->sample_rate()/2;
    float in_min_hz = in_max_hz / 4/in_stft_size;

    float out_stft_size = (in_stft_size/(float)stft->sample_rate)*block->sample_rate();

    float out_offset = (a.time - (stft->sample_offset/(float)stft->sample_rate)) * block->sample_rate();

    ::expandCompleteStft( stft->waveform_data->getCudaGlobal(),
                  block->vbo->height()->data->getCudaGlobal(),
                  out_min_hz,
                  out_max_hz,
                  out_stft_size,
                  out_offset,
                  in_min_hz,
                  in_max_hz,
                  in_stft_size,
                  0);
}


void Collection::
updateSlope( Spectrogram::pBlock block, unsigned cuda_stream )
{
    SpectrogramVbo::pHeight h = block->vbo->height();
    cudaCalculateSlopeKernel( h->data->getCudaGlobal().ptr(), block->vbo->slope()->data->getCudaGlobal().ptr(), _samples_per_block, _scales_per_block, cuda_stream );
}

void Collection::
mergeBlock( Spectrogram::pBlock outBlock, pTransform_chunk inChunk, unsigned cuda_stream, bool save_in_prepared_data)
{
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

    outBlock->isd |= InvalidSampleDescriptor( inChunk->startTime(), inChunk->endTime() );

    if (save_in_prepared_data) {
        outData->getCpuMemory();
        outBlock->prepared_data = outData;
    }
}

void Collection::
mergeBlock( Spectrogram::pBlock outBlock, Spectrogram::pBlock inBlock, unsigned cuda_stream )
{
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

    Transform::ChunkIndex end=-1;
    float in_valid_samples=samples_per_block();
    if (getNextInvalidChunk( inBlock, &end ))
        in_valid_samples = max(0.f, end*_transform->samples_per_chunk()/_transform->original_waveform()->sample_rate() - in_a.time)*in_sample_rate;
    Transform::ChunkIndex start=-1;
    if (getNextInvalidChunk( outBlock, &start )) {
        if (start>=end)
            return;
        float skipt = start*_transform->samples_per_chunk()/_transform->original_waveform()->sample_rate() - out_a.time;
        if (0<skipt) {
            in_offset += skipt*in_sample_rate;
            out_offset += skipt*out_sample_rate;
        }
    } else {
        return;
    }

    SpectrogramVbo::pHeight out_h = outBlock->vbo->height();
    SpectrogramVbo::pHeight in_h = inBlock->vbo->height();

    blockMerge( in_h->data->getCudaGlobal(),
                out_h->data->getCudaGlobal(),

                           in_sample_rate,
                           out_sample_rate,
                           in_frequency_resolution,
                           out_frequency_resolution,
                           in_offset,
                           out_offset,
                           in_valid_samples,
                           cuda_stream);

    inBlock->vbo->unmap();

    // validate block if inBlock was a valid source
    if (inBlock->ref.log2_samples_size[0] <= outBlock->ref.log2_samples_size[0] ||
        inBlock->ref.log2_samples_size[1] <= outBlock->ref.log2_samples_size[1])
    {
        if (end == (Transform::ChunkIndex)-1)
            end = out_b.time*_transform->original_waveform()->sample_rate();
        for (unsigned n=start;n<end;n++)
        {
            inBlock->valid_chunks.insert(n);
        }
    }
}

} // namespace Heightmap
