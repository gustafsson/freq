#include "collection.h"
#include "slope.cu.h"
#include "block.cu.h"
#include "blockfilter.h"

#include "tfr/cwtfilter.h"
#include "tfr/cwt.h"
#include "signal/postsink.h"

#include <boost/foreach.hpp>
#include <InvokeOnDestruction.hpp>
#include <CudaException.h>
#include <GlException.h>
#include <string>
#include <QThread>
#include <neat_math.h>
#include <debugmacros.h>
#include <Statistics.h>

#ifdef _MSC_VER
#include <msc_stdc.h>
#endif

//#define TIME_COLLECTION
#define TIME_COLLECTION if(0)

// todo remove?
#define MAX_REDUNDANT_SIZE 32

using namespace Signal;

namespace Heightmap {


///// HEIGHTMAP::COLLECTION

Collection::
        Collection( Worker* worker )
:   worker( worker ),
    _samples_per_block( 1<<7 ),
    _scales_per_block( 1<<8 ),
    _unfinished_count(0),
    _frame_counter(0),
    _postsink( new PostSink )
{

    TaskTimer tt("%s = %p", __FUNCTION__, this);
    OSVAL( tt.getStream(), _postsink->fetch_invalid_samples() ); // todo remove

    // Updated as soon as the first chunk is received
    update_sample_size( 0 );

    std::vector<pOperation> sinks;
    sinks.push_back( pOperation(new CwtToBlock(this)));
    ((PostSink*)_postsink.get())->sinks( sinks );

    _display_scale.axis_scale = Tfr::AxisScale_Logarithmic;
    _display_scale.max_frequency_scalar = 1;
    float fs = worker->source()->sample_rate();
    float minhz = Tfr::Cwt::Singleton().get_min_hz(fs);
    float maxhz = Tfr::Cwt::Singleton().get_max_hz(fs);
    _display_scale.f_min = minhz;
    _display_scale.log2f_step = log2(maxhz) - log2(minhz);
}


Collection::
        ~Collection()
{
    TaskTimer("%s = %p", __FUNCTION__, this).suppressTiming();
}


void Collection::
        reset()
{
    QMutexLocker l(&_cache_mutex);
    _cache.clear();
    _recent.clear();
}


bool Collection::
        empty()
{
    QMutexLocker l(&_cache_mutex);
    return _cache.empty() && _recent.empty();
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
    QMutexLocker l(&_cache_mutex);

    // TaskTimer tt("%s, _recent.size() = %lu", __FUNCTION__, _recent.size());

    unsigned t = _unfinished_count;
    _unfinished_count = 0;

    BOOST_FOREACH(recent_t::value_type& b, _recent)
    {
        if (b->frame_number_last_used != _frame_counter)
            break;
        b->glblock->unmap();
    }

	_frame_counter++;

    return t;
}


void Collection::
        update_sample_size( Tfr::Chunk* chunk )
{
    pOperation wf = worker->source();

    if (chunk)
    {
        _display_scale.axis_scale = Tfr::AxisScale_Logarithmic;
        _display_scale.max_frequency_scalar = 1;
        _display_scale.f_min = chunk->min_hz;
        _display_scale.log2f_step = log2(chunk->max_hz) - log2(chunk->min_hz);

        Tfr::FreqAxis fx = chunk->freqAxis();

        _min_sample_size.time = std::min( _min_sample_size.time, 1.f / chunk->sample_rate );

        // Assuming frequency resolution (in Hz, not log Hz) is the highest near 0.
        unsigned bottom_index = fx.getFrequencyIndex( _display_scale.f_min );
        float min_delta_hz = fx.getFrequency( bottom_index + 1) - fx.getFrequency( bottom_index );
        _min_sample_size.scale = std::min(
                _min_sample_size.scale,
                _display_scale.getFrequencyScalar( _display_scale.f_min  + min_delta_hz ) );
        // Old naive one: _min_sample_size.scale = 1.f/Tfr::Cwt::Singleton().nScales( FS ) );
    }
    else
    {
        _min_sample_size.time = 1.f/_samples_per_block;
        _min_sample_size.scale = 1.f/_scales_per_block;
    }

    _max_sample_size.time = std::max(_min_sample_size.time, 2.f*wf->length()/_samples_per_block);
    _max_sample_size.scale = std::max(_min_sample_size.scale, 1.f/_scales_per_block );

    // Allow for some bicubic mesh interpolation when zooming in
    _min_sample_size.scale *= 0.25;
    _min_sample_size.time *= 0.25;
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
    pOperation wf = worker->source();
    float length = wf->length();

    // Validate requested sampleSize
    sampleSize.time = fabs(sampleSize.time);
    sampleSize.scale = fabs(sampleSize.scale);

    Position minSampleSize = _min_sample_size;
    Position maxSampleSize = _max_sample_size;
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
    boost::scoped_ptr<TaskTimer> tt;
    TIME_COLLECTION
    {
        tt.reset(new TaskTimer("getBlock %s", ref.toString().c_str()));
    }

    pBlock block; // smart pointer defaults to 0
	{   QMutexLocker l(&_cache_mutex);
		cache_t::iterator itr = _cache.find( ref );
		if (itr != _cache.end())
			block = itr->second;
	}

    if (0 == block.get()) {
        block = createBlock( ref );
    } else {
        if (block->new_data_available) {
            QMutexLocker l(&block->cpu_copy_mutex);
            cudaMemcpy(block->glblock->height()->data->getCudaGlobal().ptr(),
                       block->cpu_copy->getCpuMemory(), block->cpu_copy->getNumberOfElements1D(), cudaMemcpyHostToDevice);
            block->new_data_available = false;
            computeSlope(block, 0);
        }
    }

    if (0 != block.get())
    {
        Intervals refInt = block->ref.getInterval();
        if (!(refInt-=block->valid_samples).empty())
            _unfinished_count++;

        QMutexLocker l(&_cache_mutex);

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

std::vector<pBlock> Collection::
        getIntersectingBlocks( Interval I )
{
    std::vector<pBlock> r;

    QMutexLocker l(&_cache_mutex);

    BOOST_FOREACH( cache_t::value_type& c, _cache )
    {
        pBlock& pb = c.second;
        // This check is done in mergeBlock as well, but do it here first
        // for a hopefully more local and thus faster loop.
        if (Intervals(I) & pb->ref.getInterval())
        {
            r.push_back(pb);
        }
    }

    return r;
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
        invalidate_samples( const Intervals& sid )
{
    TIME_COLLECTION TaskTimer tt("Invalidating Heightmap::Collection, %s",
                                 sid.toString().c_str());

    pOperation wf = worker->source();
    _max_sample_size.time = std::max(_max_sample_size.time, 2.f*wf->length()/_samples_per_block);

	QMutexLocker l(&_cache_mutex);
	BOOST_FOREACH( cache_t::value_type& c, _cache )
		c.second->valid_samples -= sid;
}

Intervals Collection::
        invalid_samples()
{
    Intervals r;

	QMutexLocker l(&_cache_mutex);

	BOOST_FOREACH( recent_t::value_type& b, _recent )
	{
        if (_frame_counter == b->frame_number_last_used)
        {
            Intervals i ( b->ref.getInterval() );

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
    try {
        TIME_COLLECTION TaskTimer tt("Attempt");

        pBlock attempt( new Block(ref));
        attempt->glblock.reset( new GlBlock( this ));
        {
            GlBlock::pHeight h = attempt->glblock->height();
            GlBlock::pSlope sl = attempt->glblock->slope();
        }
        attempt->glblock->unmap();

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

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

        QMutexLocker l(&_cache_mutex);
        _cache.clear();
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
                    fillBlock( block );
                    CudaException_CHECK_ERROR();
                } catch (const CudaException& x ) {
                    tt.info("Collection::createBlock, fillBlock swallowed GlException.\n%s", x.what());
                }
            }

            {
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


                // TODO compute at what log2_samples_size[1] stft is more accurate
                // than low resolution blocks.
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

        }

        if ( 0 /* set dummy values */ ) {
            GlBlock::pHeight h = block->glblock->height();
            float* p = h->data->getCpuMemory();
            for (unsigned s = 0; s<_samples_per_block/2; s++) {
                for (unsigned f = 0; f<_scales_per_block; f++) {
                    p[ f*_samples_per_block + s] = 0.05f+0.05f*sin(s*10./_samples_per_block)*cos(f*10./_scales_per_block);
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

        while (MAX_REDUNDANT_SIZE*youngest_count < _recent.size()+1 && 0<youngest_count)
        {
            Position a,b;
            _recent.back()->ref.getArea(a,b);
            TaskTimer tt("Removing block [%g:%g, %g:%g]. %u remaining blocks, recent %u blocks.", a.time, a.scale, b.time, b.scale, _cache.size(), _recent.size());

            _cache.erase(_recent.back()->ref);
            _recent.pop_back();
        }
    }

	_cache[ result->ref ] = result;

    return result;
}

#include <Statistics.h>

void Collection::
        computeSlope( pBlock block, unsigned /*cuda_stream */)
{
    TIME_COLLECTION TaskTimer tt("%s", __FUNCTION__);

    GlBlock::pHeight h = block->glblock->height();

    Position a,b;
    block->ref.getArea(a,b);

    ::cudaCalculateSlopeKernel( h->data->getCudaGlobal(),
                                block->glblock->slope()->data->getCudaGlobal(),
                                b.time-a.time, b.scale-a.scale );

    TIME_COLLECTION CudaException_ThreadSynchronize();
}


/**
  finds last source that does not have a slow operation (i.e. CwtFilter)
  among its sources.
*/
static pOperation
        fast_source(pOperation start)
{
    pOperation r = start;
    pOperation itr = start;

    while(true)
    {
        Operation* o = itr.get();
        if (!o)
            break;

        Tfr::CwtFilter* f = dynamic_cast<Tfr::CwtFilter*>(itr.get());
        if (f)
            r = f->source();

        itr = o->source();
    }

    return r;
}


void Collection::
        fillBlock( pBlock block )
{
    pOperation fast_source = Heightmap::fast_source( worker->source() );

    StftToBlock stftmerger(this);
    Tfr::Stft* transp = new Tfr::Stft();
    transp->set_approximate_chunk_size(1 << 12); // 4096
    stftmerger.transform( Tfr::pTransform( transp ));
    stftmerger.source( fast_source );
    //stftmerger.mergeChunk(block, *stft, block->glblock->height()->data);
    Tfr::ChunkAndInverse ci = stftmerger.computeChunk( block->ref.getInterval() );
    stftmerger.mergeChunk(block, *ci.chunk, block->glblock->height()->data);

    // StftToBlock (rather BlockFilter) validates samples, Discard those.
    block->valid_samples = Intervals();
}


bool Collection::
        mergeBlock( pBlock outBlock, pBlock inBlock, unsigned /*cuda_stream*/ )
{
    Interval outInterval = outBlock->ref.getInterval();

    // Find out what intervals that match
    Intervals transferDesc = inBlock->valid_samples;
    transferDesc &= outInterval;

    // Remove already computed intervals
    transferDesc -= outBlock->valid_samples;

    // If block is already up to date, abort merge
    if (transferDesc.empty())
        return false;

    TIME_COLLECTION TaskTimer tt("%s", __FUNCTION__);

    GlBlock::pHeight out_h = outBlock->glblock->height();
    GlBlock::pHeight in_h = inBlock->glblock->height();

    Position ia, ib, oa, ob;
    inBlock->ref.getArea(ia, ib);
    outBlock->ref.getArea(oa, ob);

    ::blockMerge( in_h->data->getCudaGlobal(),
                  out_h->data->getCudaGlobal(),

                  make_float4( ia.time, ia.scale, ib.time, ib.scale ),
                  make_float4( oa.time, oa.scale, ob.time, ob.scale ) );

    // Validate region of block if inBlock was source of higher resolution than outBlock
    if (inBlock->ref.log2_samples_size[0] <= outBlock->ref.log2_samples_size[0] &&
        inBlock->ref.log2_samples_size[1] <= outBlock->ref.log2_samples_size[1])
    {
        if (ib.scale>=ob.scale && ia.scale <=oa.scale)
        {
            outBlock->valid_samples |= inBlock->valid_samples;
            TIME_COLLECTION TaskTimer tt(TaskTimer::LogVerbose, "Using block %s", (inBlock->valid_samples & outInterval).toString().c_str() );
        }
    }

    //in_h.reset();

    // These inblocks won't be rendered and thus unmapped very soon. outBlock will however be unmapped
    // very soon as it was requested for rendering.
    //inBlock->glblock->unmap();

    TIME_COLLECTION CudaException_ThreadSynchronize();

    return true;
}

bool Collection::
	mergeBlock( pBlock outBlock, Reference ref, unsigned cuda_stream )
{
    // assume _cache_mutex is locked
	cache_t::iterator i = _cache.find(ref);
	if (i!=_cache.end())
		return mergeBlock(outBlock, i->second, cuda_stream );
	return false;
}

} // namespace Heightmap
