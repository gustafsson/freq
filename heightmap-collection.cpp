#include "heightmap-collection.h"
#include "heightmap-slope.cu.h"
#include "heightmap-block.cu.h"
#include "signal-filteroperation.h"
#include "tfr-cwt.h"
#include <boost/foreach.hpp>
#include <CudaException.h>
#include <GlException.h>

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
Collection( Signal::pWorker worker )
:   worker( worker ),
    _samples_per_block( 1<<7 ),
    _scales_per_block( 1<<8 )
{

}

void Collection::
    put( Signal::pBuffer b, Signal::Source* s)
{
    try {
        // Get a chunk for this block
        Tfr::pChunk chunk;

        // If buffer comes directly from a Signal::FilterOperation
        Signal::FilterOperation* filterOp = dynamic_cast<Signal::FilterOperation*>(s);
        if (filterOp) {
            // use the Cwt chunk still stored in FilterOperation
            chunk = filterOp->previous_chunk();
        } else {
            // otherwise compute the Cwt of this block
            chunk = Tfr::CwtSingleton::operate( b );
        }

        // Update all blocks with this new chunk
        BOOST_FOREACH( pBlock& pb, _cache ) {
            Position p1, p2;
            pb->ref.getArea(p1, p2);
            if (p2.time > b->start() && p1.time < b->start()+b->length())
            {
                mergeBlock( pb, chunk, 0 );
                computeSlope( pb, 0 );
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
        b->glblock->unmap();
    }

    return t;
}

Position Collection::
min_sample_size()
{
    unsigned FS = worker->source()->sample_rate();
    return Position( 1.f/FS,
                     1.f/Tfr::CwtSingleton::instance()->nScales( FS ) );
}

Position Collection::
max_sample_size()
{
    Signal::pSource wf = worker->source();
    float length = wf->length();
    Position minima=min_sample_size();

    return Position( std::max(minima.time, 2.f*length/_samples_per_block),
                     std::max(minima.scale, 1.f/_scales_per_block) );
}

/*Reference Collection::findReferenceCanonical( Position p, Position sampleSize )
{
    // doesn't ASSERT(r.containsSpectrogram() && !r.toLarge())
    Reference r(this);

    if (p.time < 0) p.time=0;
    if (p.scale < 0) p.scale=0;

    r.log2_samples_size = tvector<2,int>( floor(log2( sampleSize.time )), floor(log2( sampleSize.scale )) );
    r.block_index = tvector<2,unsigned>(p.time / _samples_per_block * pow(2, -r.log2_samples_size[0]),
                                        p.scale / _scales_per_block * pow(2, -r.log2_samples_size[1]));

    return r;
}*/

Reference Collection::
findReference( Position p, Position sampleSize )
{
    Reference r(this);

    // make sure the reference becomes valid
    Signal::pSource wf = worker->source();
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
getBlock( Reference ref )
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
            if (0 != block.get()) {
                computeSlope( block, 0 );
            }
        }

        Signal::SamplesIntervalDescriptor::Interval refInt = block->ref.getInterval();
        if (0 != block.get() &&
            (block->valid_samples &= refInt).intervals() != Signal::SamplesIntervalDescriptor( refInt ).intervals())
        {
            _unfinished_count++;
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


void Collection::
        updateInvalidSamples( Signal::SamplesIntervalDescriptor sid )
{
    BOOST_FOREACH( pBlock& b, _cache )
    {
        b->valid_samples -= sid;
    }
}

Signal::SamplesIntervalDescriptor Collection::
        getMissingSamples()
{
    Signal::SamplesIntervalDescriptor r;

    BOOST_FOREACH( pBlock& b, _cache )
    {
        Signal::SamplesIntervalDescriptor i ( b->ref.getInterval() );
        i -= b->valid_samples;
        r |= i;
    }

    return r;
}

////// private


pBlock Collection::
attempt( Reference ref )
{
    try {
        pBlock attempt( new Block(ref));
        attempt->glblock.reset( new GlBlock( this ));
        GlBlock::pHeight h = attempt->glblock->height();
        GlBlock::pSlope sl = attempt->glblock->slope();
        attempt->glblock->unmap();

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        return attempt;
    }
    catch (const CudaException& )
    { }
    catch (const GlException& )
    { }
    return pBlock();
}


pBlock Collection::
createBlock( Reference ref )
{
    Position a,b;
    ref.getArea(a,b);
    TaskTimer tt("Creating a new block [%g, %g]",a.time,b.time);
    // Try to allocate a new block
    pBlock block = attempt( ref );

    if ( 0 == block.get() && !_cache.empty()) {
        tt.info("Memory allocation failed, overwriting some older block");
        gc();
        block = attempt( ref );
    }

    if ( 0 == block.get()) {
        tt.info("Failed");
        return block; // return null-pointer
    }

    pBlock result;
    try {
        // set to zero
        GlBlock::pHeight h = block->glblock->height();
        cudaMemset( h->data->getCudaGlobal().ptr(), 0, h->data->getSizeInBytes1D() );

        if ( 1 /* create from others */ ) {
            TaskTimer tt(TaskTimer::LogVerbose, "Stubbing new block");

            // fill block by STFT
            {
                TaskTimer tt(TaskTimer::LogVerbose, "stft");
                prepareFillStft( block );
            }

            /*if (0) {
                TaskTimer tt(TaskTimer::LogVerbose, "Preventing wavelet transform");
                Position a,b;
                block->ref.getArea(a,b);
                unsigned start = a.time * worker()->source()->sample_rate();
                unsigned end = b.time * worker()->source()->sample_rate();

                for (Transform::ChunkIndex n = _transform->getChunkIndex(start);
                     n <= _transform->getChunkIndex(end);
                     n++)
                {
                    block->valid_chunks.insert( n );
                }
            }*/

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
            GlBlock::pHeight h = block->glblock->height();
            float* p = h->data->getCpuMemory();
            for (unsigned s = 0; s<_samples_per_block; s++) {
                for (unsigned f = 0; f<_scales_per_block; f++) {
                    p[ f*_samples_per_block + s] = sin(s*10./_samples_per_block)*cos(f*10./_scales_per_block);
                }
            }
        }

        result = block;
    }
    catch (const CudaException& )
    { }
    catch (const GlException& )
    { }

    if ( 0 == result.get())
        return result; // return null-pointer

    _cache.push_back( result );

    return result;
}

void Collection::
        computeSlope( pBlock block, unsigned cuda_stream )
{
    GlBlock::pHeight h = block->glblock->height();
    cudaCalculateSlopeKernel( h->data->getCudaGlobal().ptr(), block->glblock->slope()->data->getCudaGlobal().ptr(), _samples_per_block, _scales_per_block, cuda_stream );
}

void Collection::
        prepareFillStft( pBlock block )
{
    Position a, b;
    block->ref.getArea(a,b);
    float tmin = Tfr::CwtSingleton::instance()->min_hz();
    float tmax = Tfr::CwtSingleton::instance()->max_hz( worker->source()->sample_rate() );

    Tfr::Stft trans;
    Signal::pSource fast_source = Signal::Operation::fast_source( worker->source() );

    unsigned first_sample = (unsigned)(a.time*fast_source->sample_rate()),
             n_samples = (unsigned)((b.time-a.time)*fast_source->sample_rate());
    first_sample = ((first_sample-1)/trans.chunk_size+1)*trans.chunk_size;
    n_samples = ((n_samples-1)/trans.chunk_size+1)*trans.chunk_size;

    Signal::pBuffer buff = fast_source->readFixedLength( first_sample, n_samples );
    /*printf("b->number_of_samples() %% chunk_size = %d\n", buff->number_of_samples() % trans.chunk_size);
    printf("n_samples %% chunk_size = %d\n", n_samples % trans.chunk_size);
    printf("b->number_of_samples() = %d\n", buff->number_of_samples());
    printf("n_samples = %d\n", n_samples );
    printf("trans.chunk_size = %d\n", trans.chunk_size );
    fflush(stdout);*/

    Signal::pBuffer stft = trans( buff );

    float out_min_hz = exp(log(tmin) + (a.scale*(log(tmax)-log(tmin)))),
          out_max_hz = exp(log(tmin) + (b.scale*(log(tmax)-log(tmin)))),
          in_max_hz = tmax;
    float in_min_hz = in_max_hz / 4/trans.chunk_size;

    float out_stft_size = (trans.chunk_size/(float)stft->sample_rate)*block->sample_rate();

    float out_offset = (a.time - (stft->sample_offset/(float)stft->sample_rate)) * block->sample_rate();

    ::expandCompleteStft( stft->waveform_data->getCudaGlobal(),
                  block->glblock->height()->data->getCudaGlobal(),
                  out_min_hz,
                  out_max_hz,
                  out_stft_size,
                  out_offset,
                  in_min_hz,
                  in_max_hz,
                  trans.chunk_size,
                  0);
}


void Collection::
        updateSlope( pBlock block, unsigned cuda_stream )
{
    GlBlock::pHeight h = block->glblock->height();
    cudaCalculateSlopeKernel( h->data->getCudaGlobal().ptr(), block->glblock->slope()->data->getCudaGlobal().ptr(), _samples_per_block, _scales_per_block, cuda_stream );
}

void Collection::
        mergeBlock( pBlock outBlock, Tfr::pChunk inChunk, unsigned cuda_stream, bool save_in_prepared_data)
{
    boost::shared_ptr<GpuCpuData<float> > outData;

    if (!save_in_prepared_data)
        outData = outBlock->glblock->height()->data;
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
    float in_offset = std::max(0.f, (a.time-inChunk->startTime()))*in_sample_rate;
    float out_offset = std::max(0.f, (inChunk->startTime()-a.time))*out_sample_rate;

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

    outBlock->valid_samples |= Signal::SamplesIntervalDescriptor(
            (unsigned)(inChunk->startTime()*inChunk->sample_rate +.5f),
            (unsigned)(inChunk->endTime()*inChunk->sample_rate +.5f) );

    if (save_in_prepared_data) {
        outData->getCpuMemory();
        outBlock->prepared_data = outData;
    }
}

void Collection::
mergeBlock( pBlock outBlock, pBlock inBlock, unsigned cuda_stream )
{
    Signal::SamplesIntervalDescriptor in_sid = inBlock->valid_samples;
    Signal::SamplesIntervalDescriptor& out_sid = outBlock->valid_samples;
    Signal::SamplesIntervalDescriptor out_ref_sid = outBlock->ref.getInterval();
    Signal::SamplesIntervalDescriptor::Interval outInterval = out_ref_sid.intervals().front();

    // check if inBlock has any samples that can be merged with outBlock
    // by computing which samples to copy from in to out
    in_sid &= out_ref_sid; // restrict to ref block
    in_sid -= out_sid;     // remove already valid samples

    if (in_sid.intervals().empty()) {
        return;
    }

    Signal::SamplesIntervalDescriptor::Interval read_interval;
    read_interval = in_sid.getInterval(0, outInterval.last - outInterval.first );
    in_sid -= read_interval;

    float in_sample_rate = inBlock->sample_rate();
    float out_sample_rate = outBlock->sample_rate();
    unsigned signal_sample_rate = worker->source()->sample_rate();
    float in_frequency_resolution = inBlock->nFrequencies();
    float out_frequency_resolution = outBlock->nFrequencies();

    float in_offset = outInterval.first>read_interval.first?outInterval.first-read_interval.first:0;
    float out_offset = read_interval.first>outInterval.first?read_interval.first-outInterval.first:0;
    in_offset*=in_sample_rate/signal_sample_rate;
    out_offset*=out_sample_rate/signal_sample_rate;

    float in_valid_samples=read_interval.last-read_interval.first;
    in_valid_samples*=in_sample_rate/signal_sample_rate;

    GlBlock::pHeight out_h = outBlock->glblock->height();
    GlBlock::pHeight in_h = inBlock->glblock->height();

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

    inBlock->glblock->unmap();

    // validate block if inBlock was a valid source
    if (inBlock->ref.log2_samples_size[0] <= outBlock->ref.log2_samples_size[0] ||
        inBlock->ref.log2_samples_size[1] <= outBlock->ref.log2_samples_size[1])
    {
        outBlock->valid_samples |= read_interval;
    }

    // keep on going merging if there are more things to read from inBlock
    if (in_sid.intervals().size()>1) {
        mergeBlock( outBlock, inBlock, cuda_stream );
    }
}

} // namespace Heightmap
