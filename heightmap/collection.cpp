#include "collection.h"
#include "slope.cu.h"
#include "block.cu.h"
#include "signal/filteroperation.h"
#include "tfr/cwt.h"
#include <boost/foreach.hpp>
#include <InvokeOnDestruction.hpp>
#include <CudaException.h>
#include <GlException.h>
#include <string>
#include <QThread>
#include <neat_math.h>

#ifdef _MSC_VER
#include <msc_stdc.h>
#endif

//#define TIME_COLLECTION
#define TIME_COLLECTION if(0)

#define MAX_REDUNDANT_SIZE 32

namespace Heightmap {


///// HEIGHTMAP::COLLECTION

Collection::
        Collection( Signal::pWorker worker )
:   worker( worker ),
    _samples_per_block( 1<<7 ),
    _scales_per_block( 1<<8 ),
    _unfinished_count(0),
    _frame_counter(0),
    _complexInfo(ComplexInfo_Amplitude_Weighted)
{
    chunk_transform( Tfr::Cwt::SingletonP() );
}


Collection::
        ~Collection()
{
    _updates_condition.wakeAll();
    QMutexLocker l(&_updates_mutex);
    _updates.clear();
}


void Collection::
        reset()
{
	{	QMutexLocker l(&_cache_mutex);
		_cache.clear();
		_recent.clear();
	}
	{	QMutexLocker l(&_updates_mutex);
		_updates.clear();
	}
}


void Collection::
        put( Signal::pBuffer b, Signal::pSource s)
{
    Signal::SamplesIntervalDescriptor expected = expected_samples();
    if ( (expected_samples() & b->getInterval()).isEmpty() ) {
        TIME_COLLECTION TaskTimer("Collection::put received non requested block [%u, %u]", b->getInterval().first, b->getInterval().last);
        return;
    }

    TIME_COLLECTION TaskTimer tt("Collection::put [%u,%u]", b->sample_offset, b->sample_offset+b->number_of_samples());

    // Get a chunk for this block
    // The chunk might be smaller (chunk->n_valid_samples < b->number_of_samples)
    // in which case the next call to expected_samples() will return the missing
    // samples.
    Tfr::pChunk chunk = getChunk( b, s );

    if ( (expected_samples() & chunk->getInterval()).isEmpty() ) {
        TaskTimer("Collection::put received non requested chunk [%u, %u]", chunk->getInterval().first, chunk->getInterval().last);
        return;
    }

    // TODO these are filter operations, move them into filters
    switch(0){
        case 1:
        {
            float2* p     = chunk->transform_data->getCpuMemory();

            std::vector<float2> q[] = {
                std::vector<float2>(chunk->nSamples()),
                std::vector<float2>(chunk->nSamples())
            };

            unsigned bytes_per_row = sizeof(float2)*chunk->nSamples();

            float2
                    *prev_row = &q[0][0],
                    *this_row = &q[1][0];

            for (unsigned y=1; y<chunk->nScales()-1; y++)
            {
                float2* t = prev_row;
                prev_row = this_row;
                this_row = t;

                for (unsigned x=0; x<chunk->nSamples(); x++)
                {
                    float2 a = p[(y-1)*chunk->nSamples() + x];
                    float2 b = p[(y+0)*chunk->nSamples() + x];
                    float2 c = p[(y+1)*chunk->nSamples() + x];
                    float A = a.x*a.x + a.y*a.y;
                    float B = b.x*b.x + b.y*b.y;
                    float C = c.x*c.x + c.y*c.y;
                    this_row[x] = (B > A  &&  B > C) ? b : make_float2(0,0);
                }

                if (1<y)
                    memcpy( p + (y-1)*chunk->nSamples(), prev_row, bytes_per_row );
            }
            memcpy( p + (chunk->nScales()-2)*chunk->nSamples(),
                    this_row,
                    bytes_per_row );
            memset( p + (chunk->nScales()-1)*chunk->nSamples(), 0, bytes_per_row );
            memset( p                                         , 0, bytes_per_row );

            break;
        }
        case 2:
        if (1) {
            TaskTimer tt("Limited reassign");
            float2* p = chunk->transform_data->getCpuMemory();

            unsigned
                    block_size_x = 32,
                    block_size_y = 16,
                    max_dist = block_size_y/2, // Meaning nothing is allowed to move more than 8 rows away from its origin
                    // Q: What action should be taken if it wants to move further? Discard, don't move or clamp?
                    W = chunk->nSamples(),
                    H = chunk->nScales();

            std::vector<float2> q = std::vector<float2>(block_size_x*block_size_y);
            std::vector<float2> r = std::vector<float2>(block_size_x);
            std::vector<float2> r2 = std::vector<float2>(H);

            unsigned bytes_per_row = block_size_x*sizeof(float2);

            memset(&q[0], 0, bytes_per_row*block_size_y);

            unsigned c = 0;
            unsigned long long d = 0;
            for (unsigned bx=0; bx<W/block_size_x; bx++)
            {
                for (unsigned y=0; y<H; y++)
                {
                    if (max_dist <= y)
                    {
                        unsigned finished_row = (y + max_dist) % block_size_y;
                        memcpy(p + (y-max_dist)*W + bx*block_size_x, &q[finished_row * block_size_x], bytes_per_row);
                        memset(&q[finished_row * block_size_x], 0, bytes_per_row);
                    }

                    r2[y] = r[block_size_x-1];
                    memcpy(&r[0], p + y*W + bx*block_size_x, bytes_per_row);

                    for (unsigned x=0; x<block_size_x; x++)
                    {
                        float2 a;
                        if (0==x && 0==bx)
                            continue;
                        else if (0==x)
                            a = r2[y];
                        else
                            a = r[x - 1];

                        float2 b = r[x];
                        float pa = atan2(a.y, a.x);
                        float pb = atan2(b.y, b.x);
                        float FS = s->sample_rate();
                        float f = ((pb-pa)*FS)/(2*M_PI);
                        unsigned i = chunk->freqAxis().getFrequencyIndex( fabsf(f) );

                        if (i >= H)
                            i = H-1;

                        if ((i > y ? i-y : y-i) >= max_dist )
                        {
                            c ++;
                            d += (i > y ? i-y : y-i);

                            // discard
                            //continue;

                            // clamp
                            //if (i>y)
                            //    i = y + max_dist - 1;
                            //else
                            //    i = y - max_dist + 1;

                            // don't move
                            i = y;
                        }


                        q[ (i % block_size_y) * block_size_x + x].x += a.x;
                        q[ (i % block_size_y) * block_size_x + x].y += a.y;
                    }
                }

                for (unsigned y=H; y<H+max_dist; y++)
                {
                    unsigned finished_row = (y + max_dist) % block_size_y;
                    memcpy(p + (y-max_dist)*W + bx*block_size_x, &q[finished_row * block_size_x], bytes_per_row);
                    memset(&q[finished_row * block_size_x], 0, bytes_per_row);
                }
            }

            tt.info("Failed reassignments: %u, %.3g%% of all elements. Average distance: %.3g", c, c*100.f/(W*H), ((double)d)/c);

            break;
        }
        if (0) {
            TaskTimer tt("Naive reassign");
            float2* p = chunk->transform_data->getCpuMemory();

            unsigned
                    W = chunk->nSamples(),
                    H = chunk->nScales();

            std::vector<float2> q[] = {
                std::vector<float2>(H),
                std::vector<float2>(H)};

            unsigned c = 0;
            unsigned long long d = 0;

            float2
                    *prev_q = &q[0][0],
                    *this_q = &q[0][0];

            memset(&this_q[0], 0, H*sizeof(float2));

            for (unsigned x=1; x<W; x++)
            {

                float2 *T = prev_q;
                prev_q = this_q;
                this_q = T;

                memset(&this_q[0], 0, H*sizeof(float2));

                for (unsigned y=0; y<H; y++)
                {
                    float2 a = p[y*W + x - 1];
                    float2 b = p[y*W + x];
                    float pa = atan2(a.y, a.x);
                    float pb = atan2(b.y, b.x);
                    float FS = s->sample_rate();
                    float f = ((pb-pa)*FS)/(2*M_PI);
                    unsigned i = chunk->freqAxis().getFrequencyIndex( fabsf(f) );

                    this_q[ i ].x += a.x;
                    this_q[ i ].y += a.y;
                }

                for (unsigned y=0; y<H; y++)
                    p[y*W + x - 1] = prev_q[y];
            }
            for (unsigned y=0; y<H; y++)
                p[y*W + W - 1] = this_q[y];

            tt.info("Failed reassignments: %u, %.3g%% of all elements. Average distance: %.3g", c, c*100.f/(W*H), ((double)d)/c);

            break;
        }
    default:
            break;
    }

    if (_constructor_thread.isSameThread())
    {
        _updates.push_back( chunk );
        applyUpdates();
    }
    else
    {
        Tfr::pChunk cpuChunk(new Tfr::Chunk );
        cpuChunk->min_hz = chunk->min_hz;
        cpuChunk->max_hz = chunk->max_hz;
        cpuChunk->chunk_offset = chunk->chunk_offset;
        cpuChunk->sample_rate = chunk->sample_rate;
        cpuChunk->first_valid_sample = chunk->first_valid_sample;
        cpuChunk->n_valid_samples = chunk->n_valid_samples;
        cpuChunk->transform_data.reset( new GpuCpuData<float2>(0, chunk->transform_data->getNumberOfElements()));
        cudaMemcpy( cpuChunk->transform_data->getCpuMemory(),
                    chunk->transform_data->getCudaGlobal().ptr(),
                    cpuChunk->transform_data->getSizeInBytes1D(),
                    cudaMemcpyDeviceToHost );

        {
            QMutexLocker l(&_updates_mutex);
            _updates.push_back( cpuChunk );
            cpuChunk.reset(); // release cpuChunk before applyUpdate have move data to GPU
	
			_updates_condition.wait(&_updates_mutex);
		}
    }
}


void Collection::
        scales_per_block(unsigned v)
{
	QMutexLocker l(&_cache_mutex);
    _cache.clear();
	_recent.clear();
    _scales_per_block=v;
}

void Collection::
        samples_per_block(unsigned v)
{
	QMutexLocker l(&_cache_mutex);
    _cache.clear();
	_recent.clear();
    _samples_per_block=v;
}

unsigned Collection::
        next_frame()
{
    unsigned t = _unfinished_count;
    _unfinished_count = 0;

	{   QMutexLocker l(&_cache_mutex);
		BOOST_FOREACH(recent_t::value_type& b, _recent)
		{
			if (b->frame_number_last_used != _frame_counter)
				break;
			b->glblock->unmap();
		}
	}

	_frame_counter++;

    applyUpdates();
    return t;
}

Position Collection::
        min_sample_size()
{
    unsigned FS = worker->source()->sample_rate();
    return Position( 1.f/FS,
                     0.01f/Tfr::Cwt::Singleton().nScales( FS ) );
}

Position Collection::
        max_sample_size()
{
    Signal::pSource wf = worker->source();
    float length = wf->length();
    Position minima = min_sample_size();

    return Position( std::max(minima.time, 2.f*length/_samples_per_block),
                     std::max(minima.scale, 1.f/_scales_per_block) );
}

/*
  Canonical implementation of findReference

Reference Collection::findReferenceCanonical( Position p, Position sampleSize )
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
    r.log2_samples_size = tvector<2,int>( floor_log2( sampleSize.time ), floor_log2( sampleSize.scale ));
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
    r.block_index = tvector<2,unsigned>(p.time / _samples_per_block * ldexpf(1.f, -r.log2_samples_size[0]),
                                        p.scale / _scales_per_block * ldexpf(1.f, -r.log2_samples_size[1]));

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
    pBlock block; // smart pointer defaults to 0
	{   QMutexLocker l(&_cache_mutex);
		cache_t::iterator itr = _cache.find( ref );
		if (itr != _cache.end())
			block = itr->second;
	}

    if (0 == block.get()) {
        block = createBlock( ref );
    }

    if (0 != block.get())
    {
        Signal::SamplesIntervalDescriptor refInt = block->ref.getInterval();
        if (!(refInt-=block->valid_samples).isEmpty())
            _unfinished_count++;

        for( recent_t::iterator i = _recent.begin(); i!=_recent.end(); ++i )
            if ((*i)->ref == ref ) {
                _recent.erase( i );
                break;
            }

        _recent.push_front( block );

        block->frame_number_last_used = _frame_counter;
    }

    return block;
}


void Collection::
        complexInfo(ComplexInfo info)
{
    _complexInfo = info;
}


ComplexInfo Collection::
        complexInfo()
{
    return _complexInfo;
}


void Collection::
        chunk_filter( Tfr::pFilter filter )
{
    // even if (filter == _filter) then do all this, some parameter could have been changed
    // that makes the client want to reset everything

    ChunkSink::chunk_filter( filter );

    add_expected_samples( Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL );
}


void Collection::
        chunk_transform( Tfr::pTransform transform )
{
    // even if (transform == _transform) then do all this, some parameter could have been changed
    // that makes the client want to reset everything

    ChunkSink::chunk_transform( transform );

    // Remove this special case, it should be equally fast without this and more responsive
    if ( 0 != dynamic_cast<Tfr::Stft*>(transform.get()) )
    {
        QMutexLocker l(&_cache_mutex);
        _cache.clear();
    }

    add_expected_samples( Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL );
}


void Collection::
        gc()
{
	QMutexLocker l(&_cache_mutex);

	for (cache_t::iterator itr = _cache.begin(); itr!=_cache.end(); )
    {
        if (itr->second->frame_number_last_used < _frame_counter) {
            Position a,b;
            itr->second->ref.getArea(a,b);
            TaskTimer tt("Release block [%g, %g]", a.time, b.time);
			
			_recent.remove(itr->second);
            itr = _cache.erase(itr);
        } else {
            itr++;
        }
    }
}


void Collection::
        add_expected_samples( const Signal::SamplesIntervalDescriptor& sid )
{
    TIME_COLLECTION sid.print("Invalidating Heightmap::Collection");

	QMutexLocker l(&_cache_mutex);
	BOOST_FOREACH( cache_t::value_type& c, _cache )
		c.second->valid_samples -= sid;
}

Signal::SamplesIntervalDescriptor Collection::
        expected_samples()
{
    Signal::SamplesIntervalDescriptor r;

	QMutexLocker l(&_cache_mutex);

	BOOST_FOREACH( recent_t::value_type& b, _recent )
	{
        if (_frame_counter == b->frame_number_last_used)
        {
            Signal::SamplesIntervalDescriptor i ( b->ref.getInterval() );

            i -= b->valid_samples;
            r |= i;
        } else
			break;
    }

    return r;
}

////// private


pBlock Collection::
        attempt( Reference ref )
{
    TIME_COLLECTION TaskTimer tt("Attempt");
    try {
        pBlock attempt( new Block(ref));
        attempt->glblock.reset( new GlBlock( this ));
        {
            GlBlock::pHeight h = attempt->glblock->height();
            GlBlock::pSlope sl = attempt->glblock->slope();
        }
        attempt->glblock->unmap();

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        TIME_COLLECTION TaskTimer("Returning attempt").suppressTiming();
        return attempt;
    }
    catch (const CudaException& x)
    {
        /*
          Swallow silently and return null.
          createBlock will try to release old block if we're out of memory. But
          block allocation may still fail. In such a case, return null and
          heightmap::renderer will render a cross instead of this block to
          demonstrate that something went wrong. This is not a fatal error. The
          application can still continue and use filters.
          */
        TaskTimer("Collection::attempt swallowed CudaException.\n%s", x.what()).suppressTiming();
    }
    catch (const GlException& x)
    {
        // Swallow silently and return null. Same reason as 'Collection::attempt::catch (const CudaException& x)'.
        TaskTimer("Collection::attempt swallowed GlException.\n%s", x.what()).suppressTiming();
    }
    TIME_COLLECTION TaskTimer("Returning pBlock()").suppressTiming();
    return pBlock();
}


pBlock Collection::
        createBlock( Reference ref )
{
    Position a,b;
    ref.getArea(a,b);
    TIME_COLLECTION TaskTimer tt("Creating a new block [%g, %g]",a.time,b.time);
    // Try to allocate a new block
    pBlock block = attempt( ref );

	QMutexLocker l(&_cache_mutex); // Keep in scope for the remainder of this function
    if ( 0 == block.get() && !_cache.empty()) {
        TaskTimer tt("Memory allocation failed creating new block [%g, %g]. Overwriting some older block", a.time, b.time);
        l.unlock();
        gc();
        l.relock();
        block = attempt( ref );
    }

    if ( 0 == block.get()) {
        TaskTimer tt("Failed creating new block [%g, %g]", a.time, b.time);
        return pBlock(); // return null-pointer
    }

    pBlock result;
    try {
        // set to zero
        GlBlock::pHeight h = block->glblock->height();
        cudaMemset( h->data->getCudaGlobal().ptr(), 0, h->data->getSizeInBytes1D() );

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        if ( 1 /* create from others */ )
        {
            TaskTimer tt(TaskTimer::LogVerbose, "Stubbing new block");

            // fill block by STFT
            {
                TaskTimer tt(TaskTimer::LogVerbose, "stft");
                try {
                    prepareFillStft( block );
                    CudaException_CHECK_ERROR();
                } catch (const CudaException& x ) {
                    // prepareFillStft doesn't have a limit on how large
                    // waveform buffer that it tries to work on. Thus it
                    // will run out of memory when attempting to work on
                    // really large buffers.
                    tt.info("Couldn't fill new block with stft\n%s", x.what());
                }
            }

            if ( 0 != dynamic_cast<Tfr::Stft*>(_transform.get()))
            {
                block->valid_samples = block->ref.getInterval();
            } else {
                // TODO compute at what log2_samples_size[1] stft is more accurate
                // than low resolution blocks.
                if (1) {
                    TaskTimer tt(TaskTimer::LogVerbose, "Fetching details");
                    // start with the blocks that are just slightly more detailed
                    mergeBlock( block, block->ref.left(), 0 );
                    mergeBlock( block, block->ref.right(), 0 );
                    mergeBlock( block, block->ref.top(), 0 );
                    mergeBlock( block, block->ref.bottom(), 0 );
                }

                if (0) {
                    TaskTimer tt(TaskTimer::LogVerbose, "Fetching more details");
                    // then try using the blocks that are even more detailed
                    BOOST_FOREACH( cache_t::value_type& c, _cache )
                    {
                        pBlock& b = c.second;
                        if (block->ref.log2_samples_size[0] > b->ref.log2_samples_size[0] +1 ||
                            block->ref.log2_samples_size[1] > b->ref.log2_samples_size[1] +1)
                        {
                            mergeBlock( block, b, 0 );
                        }
                    }
                }

    //            GlException_CHECK_ERROR();
    //            CudaException_CHECK_ERROR();

                if (1) {
                    TaskTimer tt(TaskTimer::LogVerbose, "Fetching details");
                    // then try to upscale blocks that are just slightly less detailed
                    mergeBlock( block, block->ref.parent(), 0 );
                    mergeBlock( block, block->ref.parent().left(), 0 ); // None of these is == ref.sibbling()
                    mergeBlock( block, block->ref.parent().right(), 0 );
                    mergeBlock( block, block->ref.parent().top(), 0 );
                    mergeBlock( block, block->ref.parent().bottom(), 0 );
                }

                if (0) {
                    TaskTimer tt(TaskTimer::LogVerbose, "Fetching low resolution");
                    // then try to upscale other blocks
                    BOOST_FOREACH( cache_t::value_type& c, _cache )
                    {
                        pBlock& b = c.second;
                        if (block->ref.log2_samples_size[0] < b->ref.log2_samples_size[0]-1 ||
                            block->ref.log2_samples_size[1] < b->ref.log2_samples_size[1]-1 )
                        {
                            mergeBlock( block, b, 0 );
                        }
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

        computeSlope( block, 0 );

//        GlException_CHECK_ERROR();
//        CudaException_CHECK_ERROR();

        result = block;

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();
    }
    catch (const CudaException& x )
    {
        // Swallow silently and return null. Same reason as 'Collection::attempt::catch (const CudaException& x)'.
        TaskTimer("Collection::createBlock swallowed CudaException.\n%s", x.what()).suppressTiming();
    }
    catch (const GlException& x )
    {
        // Swallow silently and return null. Same reason as 'Collection::attempt::catch (const CudaException& x)'.
        TaskTimer("Collection::createBlock swallowed GlException.\n%s", x.what()).suppressTiming();
    }

    if ( 0 == result.get())
        return pBlock(); // return null-pointer

    if (0!= "Remove old redundant blocks")
    {
        unsigned youngest_age = -1, youngest_count = 0;
        BOOST_FOREACH( recent_t::value_type& b, _recent )
        {
            unsigned age = _frame_counter - b->frame_number_last_used;
            if (youngest_age > age) {
                youngest_age = age;
                youngest_count = 1;
            } else if(youngest_age == age) {
                ++youngest_count;
            } else {
                break; // _recent is ordered with the most recently accessed blocks first
            }
        }

        while (MAX_REDUNDANT_SIZE*youngest_count < _recent.size()+1 && 16<_recent.size())
        {
            Position a,b;
            _recent.back()->ref.getArea(a,b);
            TaskTimer tt("Removing block [%g, %g]. %u remaining blocks, recent %u blocks.", a.time, b.time, _cache.size(), _recent.size());

            _cache.erase(_recent.back()->ref);
            _recent.pop_back();
        }
    }

	_cache[ result->ref ] = result;

    return result;
}

void Collection::
        computeSlope( pBlock block, unsigned cuda_stream )
{
    TIME_COLLECTION TaskTimer tt("%s", __FUNCTION__);
    GlBlock::pHeight h = block->glblock->height();
    Position a,b;
    block->ref.getArea(a,b);
    ::cudaCalculateSlopeKernel( h->data->getCudaGlobal().ptr(), block->glblock->slope()->data->getCudaGlobal().ptr(), _samples_per_block, _scales_per_block, b.time-a.time, cuda_stream );
    TIME_COLLECTION CudaException_ThreadSynchronize();
}

// TODO prepareFillStft should be the same as putting any other chunk
void Collection::
        prepareFillStft( pBlock block )
{
    Position a, b;
    block->ref.getArea(a,b);
    float tmin = Tfr::Cwt::Singleton().min_hz();
    float tmax = Tfr::Cwt::Singleton().max_hz( worker->source()->sample_rate() );

    Tfr::Stft& trans = Tfr::Stft::Singleton();

    Signal::pSource fast_source = Signal::Operation::fast_source( worker->source() );

    unsigned first_sample = (unsigned)(a.time*fast_source->sample_rate()),
             last_sample = (unsigned)(b.time*fast_source->sample_rate()),
             margin = trans.chunk_size;
    if (first_sample >= margin)
        first_sample = ((first_sample - margin)/trans.chunk_size) * trans.chunk_size;
    else
        first_sample = 0;

    last_sample = ((last_sample + margin)/trans.chunk_size) * trans.chunk_size;

    unsigned n_samples = last_sample - first_sample;
//    first_sample = 0;
//    n_samples = (fast_source->number_of_samples()/trans.chunk_size)*trans.chunk_size;
    Signal::pBuffer buff = fast_source->readFixedLength( first_sample, n_samples );

    Tfr::pChunk stft = trans( buff );

    float out_min_hz = exp(log(tmin) + (a.scale*(log(tmax)-log(tmin)))),
          out_max_hz = exp(log(tmin) + (b.scale*(log(tmax)-log(tmin))));

    float out_stft_size = block->ref.sample_rate() / stft->sample_rate;
    float out_offset = (a.time - (stft->chunk_offset / (float)fast_source->sample_rate())) * block->ref.sample_rate();

    ::expandCompleteStft( stft->transform_data->getCudaGlobal(),
                  block->glblock->height()->data->getCudaGlobal(),
                  out_min_hz,
                  out_max_hz,
                  out_stft_size,
                  out_offset,
                  stft->min_hz,
                  stft->max_hz,
                  trans.chunk_size,
                  0);
}

void Collection::
        applyUpdates()
{
    // Make sure _updates_condition.wakeAll() is called on all exit paths.
    // This is to release the block of other threads waiting for applyUpdates to process data.
    // They should be released even if applyUpdates for some reason fails.
    InvokeOnDestrcution scopedVariable( boost::bind(&QWaitCondition::wakeAll, &_updates_condition ) );

    {   QMutexLocker l(&_updates_mutex);
        if (_updates.empty())
            return;
    }

    TIME_COLLECTION TaskTimer tt("%s", __FUNCTION__);
    TIME_COLLECTION expected_samples().print("Before apply updates");

    TIME_COLLECTION { TaskTimer t2("cudaThreadSynchronize"); CudaException_ThreadSynchronize();}

    {
        // Keep the lock while updating as a means to prevent more memory from being allocated
        QMutexLocker l(&_updates_mutex);
		QMutexLocker l2(&_cache_mutex);

        BOOST_FOREACH( Tfr::pChunk& chunk, _updates )
        {
            Signal::SamplesIntervalDescriptor chunkSid = chunk->getInterval();
            TIME_COLLECTION chunkSid.print("Applying chunk");

            // Update all blocks with this new chunk
			BOOST_FOREACH( cache_t::value_type& c, _cache )
            {
				pBlock& pb = c.second;
                // This check is done in mergeBlock as well, but do it here first
                // for a hopefully more local and thus faster loop.
                if (!(chunkSid & pb->ref.getInterval()).isEmpty())
                {
                    if (mergeBlock( &*pb, &*chunk, 0 ))
                        computeSlope( pb, 0 );
                }
                pb->glblock->unmap();
            }
        }

        _updates.clear();
    }
    TIME_COLLECTION expected_samples().print("After apply updates");

    TIME_COLLECTION CudaException_ThreadSynchronize();
}

bool Collection::
        mergeBlock( Block* outBlock, Tfr::Chunk* inChunk, unsigned cuda_stream, bool save_in_prepared_data)
{
    // Find out what intervals that match
    Signal::SamplesIntervalDescriptor::Interval outInterval = outBlock->ref.getInterval();
    Signal::SamplesIntervalDescriptor::Interval inInterval = inChunk->getInterval();

    Signal::SamplesIntervalDescriptor transferDesc = inInterval;
    transferDesc &= outInterval;

    // Remove already computed intervals
    transferDesc -= outBlock->valid_samples;

    // If block is already up to date, abort merge
    if (transferDesc.isEmpty())
        return false;

    std::stringstream ss;
    Position a,b;
    outBlock->ref.getArea(a,b);
    TIME_COLLECTION TaskTimer tt("%s chunk t=[%g, %g[ into block t=[%g,%g] ff=[%g,%g]%s", __FUNCTION__,
                                 inChunk->startTime(), inChunk->endTime(), a.time, b.time, a.scale, b.scale,
                                 save_in_prepared_data?" save_in_prepared_data":"");

    boost::shared_ptr<GpuCpuData<float> > outData;

    // If mergeBlock is called by a separate worker thread it will also have a separate cuda context
    // and thus cannot write directly to the cuda buffer that is mapped to the rendering thread's
    // OpenGL buffer. Instead merge inChunk to a prepared_data buffer in outBlock now, move data to CPU
    // and update the OpenGL block the next time it is rendered.
    if (!save_in_prepared_data) {
        outData = outBlock->glblock->height()->data;
    } else {
        if (outBlock->prepared_data)
            outData = outBlock->prepared_data;
        else
            outData.reset( new GpuCpuData<float>(0, make_cudaExtent( _samples_per_block, _scales_per_block, 1 ), GpuCpuVoidData::CudaGlobal ) );
    }

    float in_sample_rate = inChunk->sample_rate;
    float out_sample_rate = outBlock->ref.sample_rate();
    float in_frequency_resolution = inChunk->nScales();
    float out_frequency_resolution = outBlock->ref.nFrequencies();

    BOOST_FOREACH( Signal::SamplesIntervalDescriptor::Interval transfer, transferDesc.intervals())
    {
        TIME_COLLECTION TaskTimer tt("Inserting chunk [%u,%u]", transfer.first, transfer.last);

        if (0)
        TIME_COLLECTION {
            TaskTimer("inInterval [%u,%u]", inInterval.first, inInterval.last).suppressTiming();
            TaskTimer("outInterval [%u,%u]", outInterval.first, outInterval.last).suppressTiming();
            TaskTimer("inChunk->first_valid_sample = %u", inChunk->first_valid_sample).suppressTiming();
            TaskTimer("inChunk->n_valid_samples = %u", inChunk->n_valid_samples).suppressTiming();
        }

        // Find resolution and offest for the two blocks to be merged
        float in_sample_offset = transfer.first - inInterval.first;
        float out_sample_offset = transfer.first - outInterval.first;

        // Add offset for redundant samples in inChunk
        in_sample_offset += inChunk->first_valid_sample;

        // Rescale out_offset to out_sample_rate (samples are originally
        // described in the inChunk sample rate)
        out_sample_offset *= out_sample_rate / in_sample_rate;

        float in_frequency_offset = a.scale * in_frequency_resolution;
        float out_frequency_offset = 0;

        TaskTimer("a.scale = %g", a.scale).suppressTiming();
        TaskTimer("in_frequency_resolution = %g", in_frequency_resolution).suppressTiming();
        TaskTimer("out_frequency_resolution = %g", out_frequency_resolution).suppressTiming();

        // Invoke CUDA kernel execution to merge blocks
        ::blockMergeChunk( inChunk->transform_data->getCudaGlobal(),
                           outData->getCudaGlobal(),

                           in_sample_rate,
                           out_sample_rate,
                           in_frequency_resolution,
                           out_frequency_resolution,
                           in_sample_offset,
                           out_sample_offset,
                           in_frequency_offset,
                           out_frequency_offset,
                           transfer.last - transfer.first,
                           _complexInfo,
                           cuda_stream);

        outBlock->valid_samples |= transfer;
        TIME_COLLECTION CudaException_ThreadSynchronize();
    }

    if (save_in_prepared_data) {
        outData->getCpuMemory();
        outData->freeUnused();
        outBlock->prepared_data = outData;
    }

    TIME_COLLECTION CudaException_ThreadSynchronize();
    return true;
}

bool Collection::
        mergeBlock( pBlock outBlock, pBlock inBlock, unsigned cuda_stream )
{
    Signal::SamplesIntervalDescriptor::Interval inInterval = inBlock->ref.getInterval();
    Signal::SamplesIntervalDescriptor::Interval outInterval = outBlock->ref.getInterval();

    // Find out what intervals that match
    Signal::SamplesIntervalDescriptor transferDesc = inBlock->valid_samples;
    transferDesc &= outInterval;

    // Remove already computed intervals
    transferDesc -= outBlock->valid_samples;

    // If block is already up to date, abort merge
    if (transferDesc.isEmpty())
        return false;

    TIME_COLLECTION TaskTimer tt("%s", __FUNCTION__);

    float in_sample_rate = inBlock->ref.sample_rate();
    float out_sample_rate = outBlock->ref.sample_rate();
    unsigned signal_sample_rate = worker->source()->sample_rate();
    float in_frequency_resolution = inBlock->ref.nFrequencies();
    float out_frequency_resolution = outBlock->ref.nFrequencies();

    GlBlock::pHeight out_h = outBlock->glblock->height();
    GlBlock::pHeight in_h = inBlock->glblock->height();

    BOOST_FOREACH( const Signal::SamplesIntervalDescriptor::Interval& transfer, transferDesc.intervals())
    {
        float in_offset = transfer.first - inInterval.first;
        float out_offset = transfer.first - outInterval.first;
        float in_valid_samples = transfer.last - transfer.first;

        // Rescale to proper sample rates (samples are originally
        // described in the signal_sample_rate)
        in_offset *= in_sample_rate / signal_sample_rate;
        out_offset *= out_sample_rate / signal_sample_rate;
        in_valid_samples *= in_sample_rate / signal_sample_rate;

        ::blockMerge( in_h->data->getCudaGlobal(),
                      out_h->data->getCudaGlobal(),

                      in_sample_rate,
                      out_sample_rate,
                      in_frequency_resolution,
                      out_frequency_resolution,
                      in_offset,
                      out_offset,
                      in_valid_samples,
                      cuda_stream);

        // Validate region of block if inBlock was source of higher resolution than outBlock
        if (inBlock->ref.log2_samples_size[0] <= outBlock->ref.log2_samples_size[0] &&
            inBlock->ref.log2_samples_size[1] <= outBlock->ref.log2_samples_size[1])
        {
            outBlock->valid_samples |= transfer;
            TIME_COLLECTION TaskTimer tt(TaskTimer::LogVerbose, "Using block [%u,%u]", transfer.first, transfer.last);
        }
    }

    in_h.reset();
    out_h.reset();

    // These inblocks won't be rendered and thus unmapped very soon. outBlock will however be unmapped
    // very soon as it was requested for rendering.
    inBlock->glblock->unmap();

    TIME_COLLECTION CudaException_ThreadSynchronize();

    return true;
}

bool Collection::
	mergeBlock( pBlock outBlock, Reference ref, unsigned cuda_stream )
{
	cache_t::iterator i = _cache.find(ref);
	if (i!=_cache.end())
		return mergeBlock(outBlock, i->second, cuda_stream );
	return false;
}

} // namespace Heightmap
