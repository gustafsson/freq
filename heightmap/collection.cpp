#include "collection.h"
#include "block.cu.h"
#include "blockfilter.h"

#include "tfr/cwtfilter.h"
#include "tfr/cwt.h"
#include "signal/operationcache.h"

#include <InvokeOnDestruction.hpp>
#include <CudaException.h>
#include <GlException.h>
#include <string>
#include <neat_math.h>
#include <debugmacros.h>
#include <Statistics.h>

#include <msc_stdc.h>

#define TIME_COLLECTION
//#define TIME_COLLECTION if(0)

#define VERBOSE_COLLECTION
//#define VERBOSE_COLLECTION if(0)

//#define VERBOSE_EACH_FRAME_COLLECTION
#define VERBOSE_EACH_FRAME_COLLECTION if(0)

// #define TIME_GETBLOCK
#define TIME_GETBLOCK if(0)

// Don't keep more than this times the number of blocks currently needed
// TODO define this as fraction of total memory instead using cacheByteSize
#define MAX_REDUNDANT_SIZE 80
#define MAX_CREATED_BLOCKS_PER_FRAME 2 // Even numbers look better for stereo signals

using namespace Signal;

namespace Heightmap {


///// HEIGHTMAP::COLLECTION

Collection::
        Collection( pOperation target )
:   target( target ),
    _samples_per_block( 1<<7 ), // Created for each
    _scales_per_block( 1<<8 ),
    _unfinished_count(0),
    _created_count(0),
    _frame_counter(0)
{
        BOOST_ASSERT( target );

    TaskTimer tt("%s = %p", __FUNCTION__, this);

    // Updated as soon as the first chunk is received
    update_sample_size( 0 );

//    _display_scale.axis_scale = Tfr::AxisScale_Logarithmic;
    _display_scale.axis_scale = Tfr::AxisScale_Linear;
    _display_scale.max_frequency_scalar = 1;
    float fs = target->sample_rate();
    float minhz = Tfr::Cwt::Singleton().get_min_hz(fs);
    float maxhz = Tfr::Cwt::Singleton().get_max_hz(fs);
    _display_scale.min_hz = minhz;
    //_display_scale.log2f_step = log2(maxhz) - log2(minhz);
    _display_scale.min_hz = 0;
    _display_scale.f_step = maxhz - minhz;
}


Collection::
        ~Collection()
{
    TaskInfo("%s = %p", __FUNCTION__, this);
}


void Collection::
        reset()
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif
    _cache.clear();
    _recent.clear();
}


bool Collection::
        empty()
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif
    return _cache.empty() && _recent.empty();
}


void Collection::
        scales_per_block(unsigned v)
{
#ifndef SAWE_NO_MUTEX
	QMutexLocker l(&_cache_mutex);
#endif
    _cache.clear();
	_recent.clear();
    _scales_per_block=v;
}


void Collection::
        samples_per_block(unsigned v)
{
#ifndef SAWE_NO_MUTEX
	QMutexLocker l(&_cache_mutex);
#endif
    _cache.clear();
	_recent.clear();
    _samples_per_block=v;
}


unsigned Collection::
        next_frame()
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif

    // TaskTimer tt("%s, _recent.size() = %lu", __FUNCTION__, _recent.size());

    boost::scoped_ptr<TaskTimer> tt;
    if (_unfinished_count)
    {
        VERBOSE_EACH_FRAME_COLLECTION tt.reset( new TaskTimer("Collection::next_frame(), %u, %u", _unfinished_count, _created_count));
    }
        // TaskTimer tt("%s, _recent.size() = %lu", __FUNCTION__, _recent.size());

    unsigned t = _unfinished_count;
    _unfinished_count = 0;
    _created_count = 0;


    /*foreach(const recent_t::value_type& b, _recent)
    {
        if (b->frame_number_last_used != _frame_counter)
            break;
        b->glblock->unmap();
    }*/
    foreach(const cache_t::value_type& b, _cache)
    {
        b.second->glblock->unmap();

        if (b.second->frame_number_last_used != _frame_counter)
        {
            b.second->glblock->delete_texture();
        }
    }

	_frame_counter++;

    return t;
}


void Collection::
        update_sample_size( Tfr::Chunk* chunk )
{
    if (chunk)
    {
        //_display_scale.axis_scale = Tfr::AxisScale_Logarithmic;
        //_display_scale.max_frequency_scalar = 1;
        //_display_scale.f_min = chunk->min_hz;
        //_display_scale.log2f_step = log2(chunk->max_hz) - log2(chunk->min_hz);


        _min_sample_size.time = std::min( _min_sample_size.time, 0.25f / chunk->sample_rate );
        _min_sample_size.time = std::min( _min_sample_size.time, 0.25f / chunk->original_sample_rate );
        //TaskInfo("_min_sample_size.time = %g", _min_sample_size.time);

        if (chunk->transform_data)
        {
            Tfr::FreqAxis fx = chunk->freqAxis;
            unsigned top_index = fx.getFrequencyIndex( _display_scale.getFrequency(1.f) ) - 1;
            _min_sample_size.scale = std::min(
                    _min_sample_size.scale,
                    (1 - _display_scale.getFrequencyScalar( fx.getFrequency( top_index )))*0.01f);

            // Old naive one:
            //_min_sample_size.scale = std::min(
            //        _min_sample_size.scale,
            //        1.f/Tfr::Cwt::Singleton().nScales( chunk->sample_rate ) );
        }
    }
    else
    {
        _min_sample_size.time = 1.f/_samples_per_block;
        _min_sample_size.scale = 1.f/_scales_per_block;
        // Allow for some bicubic mesh interpolation when zooming in
        _min_sample_size.scale *= 0.25f;
        _min_sample_size.time *= 0.25f;
    }

    _max_sample_size.time = std::max(_min_sample_size.time, 2.f*target->length()/_samples_per_block);
    _max_sample_size.scale = std::max(_min_sample_size.scale, 1.f/_scales_per_block );
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
    float length = target->length();

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
    TIME_GETBLOCK TaskTimer tt("getBlock %s", ref.toString().c_str());

    pBlock block; // smart pointer defaults to 0
    {   
		#ifndef SAWE_NO_MUTEX
		QMutexLocker l(&_cache_mutex);
		#endif
        cache_t::iterator itr = _cache.find( ref );
        if (itr != _cache.end())
        {
            block = itr->second;
        }
    }

    if (0 == block.get()) {
        if ( MAX_CREATED_BLOCKS_PER_FRAME > _created_count )
        {
            block = createBlock( ref );
            _created_count++;
        }
    } else {
			#ifndef SAWE_NO_MUTEX
        if (block->new_data_available) {
            QMutexLocker l(&block->cpu_copy_mutex);
            cudaMemcpy(block->glblock->height()->data->getCudaGlobal().ptr(),
                       block->cpu_copy->getCpuMemory(), block->cpu_copy->getNumberOfElements1D(), cudaMemcpyHostToDevice);
            block->new_data_available = false;
        }
			#endif
    }

    if (0 != block.get())
    {
        Intervals refInt = block->ref.getInterval();
        if (!(refInt-=block->valid_samples).empty())
            _unfinished_count++;

		#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_cache_mutex);
		#endif

        _recent.remove( block );
        _recent.push_front( block );

        block->frame_number_last_used = _frame_counter;
    }

    return block;
}

std::vector<pBlock> Collection::
        getIntersectingBlocks( Interval I, bool only_visible )
{
    std::vector<pBlock> r;
    r.reserve(8);

	#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
	#endif
    //TIME_COLLECTION TaskTimer tt("getIntersectingBlocks( %s, %s ) from %u caches", I.toString().c_str(), only_visible?"only visible":"all", _cache.size());

    foreach ( const cache_t::value_type& c, _cache )
    {
        const pBlock& pb = c.second;
        
        if (only_visible && _frame_counter != pb->frame_number_last_used)
            continue;

        // This check is done in mergeBlock as well, but do it here first
        // for a hopefully more local and thus faster loop.
        if (I.isConnectedTo(pb->ref.getInterval()))
        {
            r.push_back(pb);
        }
    }

/*    // consistency check
    foreach( const cache_t::value_type& c, _cache )
    {
        bool found = false;
        foreach( const recent_t::value_type& r, _recent )
        {
            if (r == c.second)
                found = true;
        }

        BOOST_ASSERT(found);
    }

    // consistency check
    foreach( const recent_t::value_type& r, _recent )
    {
        bool found = false;
        foreach( const cache_t::value_type& c, _cache )
        {
            if (r == c.second)
                found = true;
        }

        BOOST_ASSERT(found);
    }*/

    return r;
}


unsigned long Collection::
        cacheByteSize()
{
    // For each block there may be both a slope map and heightmap. Also there
    // may be both a texture and a vbo, and possibly a mapped cuda copy.
    //
    // But most of the blocks will only have a heightmap vbo, and none of the
    // others. It would be possible to look through the list and into each
    // cache block to see what data is allocated at the momement. For a future
    // release perhaps...
    //unsigned long estimation = _cache.size() * scales_per_block()*samples_per_block()*1*sizeof(float)*2;
    //return estimation;

    unsigned long sumsize = 0;
    foreach (const cache_t::value_type& b, _cache)
    {
        sumsize += b.second->glblock->allocated_bytes_per_element();
    }

    unsigned elements_per_block = scales_per_block()*samples_per_block();
    return sumsize*elements_per_block;
}


unsigned Collection::
        cacheCount()
{
    return _cache.size();
}


void Collection::
        printCacheSize()
{
    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);
    float MB = 1./1024/1024;
    TaskInfo("Currently has %u cached blocks (ca %g MB). There are %g MB graphics memory free of a total of %g MB",
             _cache.size(), cacheByteSize() * MB, free * MB, total * MB );
}


void Collection::
        gc()
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif

    TaskTimer tt("Collection doing garbage collection", _cache.size());
    printCacheSize();
    TaskInfo("Of which %u are recently used", _recent.size());

    for (cache_t::iterator itr = _cache.begin(); itr!=_cache.end(); )
    {
        if (_frame_counter != itr->second->frame_number_last_used ) {
            Position a,b;
            itr->second->ref.getArea(a,b);
            TaskTimer tt("Release block [%g, %g]", a.time, b.time);

            _recent.remove(itr->second);
            itr = _cache.erase(itr);
        } else {
            itr++;
        }
    }

    TaskInfo("Now has %u cached blocks (ca %g MB)", _cache.size(),
             _cache.size() * scales_per_block()*samples_per_block()*(1+2)*sizeof(float)*1e-6 );
}


void Collection::
        display_scale(Tfr::FreqAxis a)
{
    _display_scale = a;
    invalidate_samples( target->getInterval() );
}


void Collection::
        invalidate_samples( const Intervals& sid )
{
    TIME_COLLECTION TaskTimer tt("Invalidating Heightmap::Collection, %s",
                                 sid.toString().c_str());

    _max_sample_size.time = std::max(_max_sample_size.time, 2.f*target->length()/_samples_per_block);

#ifndef SAWE_NO_MUTEX
	QMutexLocker l(&_cache_mutex);
#endif
    foreach ( const cache_t::value_type& c, _cache )
		c.second->valid_samples -= sid;
}

Intervals Collection::
        invalid_samples()
{
    Intervals r;

#ifndef SAWE_NO_MUTEX
	QMutexLocker l(&_cache_mutex);
#endif

    unsigned counter = 0;
    {
    //TIME_COLLECTION TaskTimer tt("Collection::invalid_samples, %u, %p", _recent.size(), this);

    foreach ( const recent_t::value_type& b, _recent )
    {
        if (_frame_counter == b->frame_number_last_used)
        {
            counter++;
            Intervals i = b->ref.getInterval();

            i -= b->valid_samples;

            r |= i;

            VERBOSE_EACH_FRAME_COLLECTION
            if (i)
                TaskInfo("block %s is invalid on %s", b->ref.toString().c_str(), i.toString().c_str());
        } else
            break;
    }
    }

    //TIME_COLLECTION TaskInfo("%u blocks with invalid samples %s", counter, r.toString().c_str());

    return r;
}

////// private


pBlock Collection::
        attempt( Reference ref )
{
    try {
        TIME_COLLECTION TaskTimer tt("Allocation attempt");

        pBlock attempt( new Block(ref));
        Position a,b;
        ref.getArea(a,b);
        attempt->glblock.reset( new GlBlock( this, b.time-a.time, b.scale-a.scale ));
/*        {
            GlBlock::pHeight h = attempt->glblock->height();
            //GlBlock::pSlope sl = attempt->glblock->slope();
        }
        attempt->glblock->unmap();
*/
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
        TaskInfo tt("Collection::attempt swallowed CudaException.\n%s", x.what());
        printCacheSize();
    }
    catch (const GlException& x)
    {
        // Swallow silently and return null. Same reason as 'Collection::attempt::catch (const CudaException& x)'.
        TaskInfo("Collection::attempt swallowed GlException.\n%s", x.what());
    }
    return pBlock();
}


pBlock Collection::
        createBlock( Reference ref )
{
    Position a,b;
    ref.getArea(a,b);
    TIME_COLLECTION TaskTimer tt("Creating a new block [(%g %g), (%g %g)]",a.time, a.scale, b.time, b.scale);
    // Try to allocate a new block
    pBlock result;
    try
    {
        pBlock block = attempt( ref );

		bool empty_cache;
		{
#ifndef SAWE_NO_MUTEX
            QMutexLocker l(&_cache_mutex);
#endif
			empty_cache = _cache.empty();
		}

        if ( 0 == block.get() && !empty_cache) {
            TaskTimer tt("Memory allocation failed creating new block [%g, %g]. Doing garbage collection", a.time, b.time);
            gc();
            block = attempt( ref );
        }
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_cache_mutex); // Keep in scope for the remainder of this function
#endif

        if ( 0 == block.get()) {
            TaskTimer tt("Failed creating new block [%g, %g]", a.time, b.time);
            return pBlock(); // return null-pointer
        }

        // set to zero
        GlBlock::pHeight h = block->glblock->height();
        cudaMemset( h->data->getCudaGlobal().ptr(), 0, h->data->getSizeInBytes1D() );

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        bool tfr_is_stft = false;
        Tfr::Filter* filter = dynamic_cast<Tfr::Filter*>(_filter.get());
        if (filter)
            tfr_is_stft = dynamic_cast<Tfr::Stft*>(filter->transform().get());

        if ( 1 /* create from others */ )
        {
            VERBOSE_COLLECTION TaskTimer tt("Stubbing new block");

            // fill block by STFT during the very first frames
            if (!tfr_is_stft) // don't stubb if they will be filled by stft shortly hereafter
            if (10 > _frame_counter || true) {
                //size_t stub_size
                //if ((b.time - a.time)*fast_source( target)->sample_rate()*sizeof(float) )
                try
                {
                    TIME_COLLECTION TaskTimer tt("stft");

                    fillBlock( block );
                    CudaException_CHECK_ERROR();
                }
                catch (const CudaException& x )
                {
                    // Swallow silently, it is not fatal if this stubbed fft can't be computed right away
                    TaskInfo tt("Collection::fillBlock swallowed CudaException.\n%s", x.what());
                    printCacheSize();
                }
            }

            {
                if (1) {
                    TIME_COLLECTION TaskTimer tt("Fetching data from others");
                    // then try to upscale other blocks
                    Intervals things_to_update = ref.getInterval();
                    /*Intervals all_things;
                    foreach( const cache_t::value_type& c, _cache )
                    {
                        all_things |= c.second->ref.getInterval();
                    }
                    things_to_update &= all_things;*/

                    //for (int dist = 10; dist<-10; --dist)
                    {
#ifndef SAWE_NO_MUTEX
                        l.unlock();
#endif
                        std::vector<pBlock> gib = getIntersectingBlocks( things_to_update, false );
#ifndef SAWE_NO_MUTEX
                        l.relock();
#endif
                        TIME_COLLECTION TaskTimer tt("Looping");
                        foreach( const pBlock& bl, gib )
                        {
                            Interval v = bl->ref.getInterval();
                            if ( !(things_to_update & v ))
                                continue;

                            int d = bl->ref.log2_samples_size[0];
                            d -= ref.log2_samples_size[0];
                            d += bl->ref.log2_samples_size[1];
                            d -= ref.log2_samples_size[1];

                            //if (d==dist)
                            //mergeBlock( block, bl, 0 );
                            if (d>=-2 && d<=2)
                            {
                                Position a2,b2;
                                bl->ref.getArea(a2,b2);
                                if (a2.scale <= a.scale && b2.scale >= b.scale )
                                {
                                    things_to_update -= v;
                                    mergeBlock( block, bl, 0 );
                                }
                                else if (bl->ref.log2_samples_size[1] + 1 == ref.log2_samples_size[1])
                                {
                                    mergeBlock( block, bl, 0 );
                                }
                            }
                        }
                    }
                }

                if (0) {
                    VERBOSE_COLLECTION TaskTimer tt("Fetching low resolution");
                    // then try to upscale other blocks
                    foreach( const cache_t::value_type& c, _cache )
                    {
                        const pBlock& b = c.second;
                        if (block->ref.log2_samples_size[0] < b->ref.log2_samples_size[0]-1 ||
                            block->ref.log2_samples_size[1] < b->ref.log2_samples_size[1]-1 )
                        {
                            mergeBlock( block, b, 0 );
                        }
                    }
                }


                // TODO compute at what log2_samples_size[1] stft is more accurate
                // than low resolution blocks. So that Cwt is not needed.
                if (0) {
                    VERBOSE_COLLECTION TaskTimer tt("Fetching details");
                    // then try to upscale blocks that are just slightly less detailed
                    foreach( const cache_t::value_type& c, _cache )
                    {
                        const pBlock& b = c.second;
                        if (block->ref.log2_samples_size[0] == b->ref.log2_samples_size[0] &&
                            block->ref.log2_samples_size[1]+1 == b->ref.log2_samples_size[1])
                        {
                            mergeBlock( block, b, 0 );
                        }
                    }
                    foreach( const cache_t::value_type& c, _cache )
                    {
                        const pBlock& b = c.second;
                        if (block->ref.log2_samples_size[0]+1 == b->ref.log2_samples_size[0] &&
                            block->ref.log2_samples_size[1] == b->ref.log2_samples_size[1])
                        {
                            mergeBlock( block, b, 0 );
                        }
                    }
                }

                if (0) {
                    VERBOSE_COLLECTION TaskTimer tt("Fetching more details");
                    // then try using the blocks that are even more detailed
                    foreach( const cache_t::value_type& c, _cache )
                    {
                        const pBlock& b = c.second;
                        if (block->ref.log2_samples_size[0] > b->ref.log2_samples_size[0] +1 ||
                            block->ref.log2_samples_size[1] > b->ref.log2_samples_size[1] +1)
                        {
                            mergeBlock( block, b, 0 );
                        }
                    }
                }

                if (0) {
                    VERBOSE_COLLECTION TaskTimer tt("Fetching details");
                    // start with the blocks that are just slightly more detailed
                    foreach( const cache_t::value_type& c, _cache )
                    {
                        const pBlock& b = c.second;
                        if (block->ref.log2_samples_size[0] == b->ref.log2_samples_size[0]  &&
                            block->ref.log2_samples_size[1] == b->ref.log2_samples_size[1] +1)
                        {
                            mergeBlock( block, b, 0 );
                        }
                    }
                    foreach( const cache_t::value_type& c, _cache )
                    {
                        const pBlock& b = c.second;
                        if (block->ref.log2_samples_size[0] == b->ref.log2_samples_size[0] +1 &&
                            block->ref.log2_samples_size[1] == b->ref.log2_samples_size[1] )
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

        result = block;

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();
    }
    catch (const CudaException& x )
    {
        // Swallow silently and return null. Same reason as 'Collection::attempt::catch (const CudaException& x)'.
        TaskInfo("Collection::createBlock swallowed CudaException.\n%s", x.what());
        printCacheSize();
        return pBlock();
    }
    catch (const GlException& x )
    {
        // Swallow silently and return null. Same reason as 'Collection::attempt::catch (const CudaException& x)'.
        TaskTimer("Collection::createBlock swallowed GlException.\n%s", x.what()).suppressTiming();
        return pBlock();
    }

    BOOST_ASSERT( 0 != result.get() );

    {TaskTimer t2("Removing old redundant blocks");
    if (0!= "Remove old redundant blocks")
    {
        unsigned youngest_age = -1, youngest_count = 0;
        foreach ( const recent_t::value_type& b, _recent )
        {
            unsigned age = _frame_counter - b->frame_number_last_used;
            if (youngest_age > age) {
                youngest_age = age;
                youngest_count = 1;
            } else if (youngest_age == age) {
                ++youngest_count;
            } else {
                break; // _recent is ordered with the most recently accessed blocks first
            }
        }

        while (MAX_REDUNDANT_SIZE*youngest_count < _recent.size() && 0<youngest_count)
        {
            Position a,b;
            _recent.back()->ref.getArea(a,b);
            TaskTimer tt("Removing block [%g:%g, %g:%g]. %u remaining blocks, recent %u blocks.", a.time, a.scale, b.time, b.scale, _cache.size(), _recent.size());

            _cache.erase(_recent.back()->ref);
            _recent.pop_back();
        }
    }
    }

    // result is non-zero
    _cache[ result->ref ] = result;

    //TIME_COLLECTION printCacheSize();

    TIME_COLLECTION CudaException_ThreadSynchronize();

    return result;
}


/**
  finds a source that we reliably can read from fast
*/
static pOperation
        fast_source(pOperation start, Interval I)
{
    pOperation itr = start;

    if (!itr)
        return itr;

    while (true)
    {
        OperationCache* c = dynamic_cast<OperationCache*>( itr.get() );
        if (c)
        {
            if ((Intervals(I) - c->cached_samples()).empty())
            {
                return itr;
            }
        }

        pOperation s = itr->source();
        if (!s)
            return itr;
        
        itr = s;
    }
}


void Collection::
        fillBlock( pBlock block )
{
    StftToBlock stftmerger(this);
    Tfr::Stft* transp;
    stftmerger.transform( Tfr::pTransform( transp = new Tfr::Stft() ));
    transp->set_approximate_chunk_size(1 << 12); // 4096
    stftmerger.exclude_end_block = true;

    // Only take 4 MB of signal data at a time
    unsigned section_size = (4<<20) / sizeof(float);
    Intervals sections = block->ref.getInterval();
    sections &= Interval(0, target->number_of_samples());

    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::ptime start_time = now;
    unsigned time_to_work_ms = 20; // should count as interactive
    while (sections)
    {
        Interval section = sections.fetchInterval(section_size);
        Interval needed = stftmerger.requiredInterval( section );
        pOperation fast_source = Heightmap::fast_source( target, needed );
        stftmerger.source( fast_source );
        Tfr::ChunkAndInverse ci = stftmerger.computeChunk( section );
        stftmerger.mergeChunk(block, *ci.chunk, block->glblock->height()->data);
        Interval chunk_interval = ci.chunk->getInterval();

        sections -= chunk_interval;

        now = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration diff = now - start_time;
        // Don't bother creating stubbed blocks for more than a fraction of a second
        if (diff.total_milliseconds() > time_to_work_ms)
            break;
    }

    // StftToBlock (rather BlockFilter) validates samples, Discard those.
    block->valid_samples = Intervals();
}


bool Collection::
        mergeBlock( pBlock outBlock, pBlock inBlock, unsigned /*cuda_stream*/ )
{
    const Interval outInterval = outBlock->ref.getInterval();

    // Find out what intervals that match
    Intervals transferDesc = inBlock->ref.getInterval();
    transferDesc &= outInterval;

    // Remove already computed intervals
    transferDesc -= outBlock->valid_samples;

    // If block is already up to date, abort merge
    if (transferDesc.empty())
        return false;

    Position ia, ib, oa, ob;
    inBlock->ref.getArea(ia, ib);
    outBlock->ref.getArea(oa, ob);

    if (ia.scale >= ob.scale || ib.scale<=oa.scale)
        return false;

    TIME_COLLECTION TaskTimer tt("%s, %s into %s", __FUNCTION__,
                                 inBlock->ref.toString().c_str(), outBlock->ref.toString().c_str());

    GlBlock::pHeight out_h = outBlock->glblock->height();
    GlBlock::pHeight in_h = inBlock->glblock->height();

    BOOST_ASSERT( in_h.get() != out_h.get() );
    BOOST_ASSERT( outBlock.get() != inBlock.get() );

#ifdef CUDA_MEMCHECK_TEST
    Block::pData copy( new GpuCpuData<float>( *out_h->data ));
    out_h->data.swap( copy );
#endif

    ::blockMerge( in_h->data->getCudaGlobal(),
                  out_h->data->getCudaGlobal(),

                  make_float4( ia.time, ia.scale, ib.time, ib.scale ),
                  make_float4( oa.time, oa.scale, ob.time, ob.scale ) );

#ifdef CUDA_MEMCHECK_TEST
    out_h->data.swap( copy );
    *out_h->data = *copy;
#endif

    // Validate region of block if inBlock was source of higher resolution than outBlock
    if (0)
    if (inBlock->ref.log2_samples_size[0] <= outBlock->ref.log2_samples_size[0] &&
        inBlock->ref.log2_samples_size[1] <= outBlock->ref.log2_samples_size[1])
    {
        if (ib.scale>=ob.scale && ia.scale <=oa.scale)
        {
            outBlock->valid_samples -= inBlock->ref.getInterval();
            outBlock->valid_samples |= inBlock->valid_samples;
            outBlock->valid_samples &= outInterval;
            VERBOSE_COLLECTION TaskTimer tt("Using block %s", (inBlock->valid_samples & outInterval).toString().c_str() );
        }
    }

    TIME_COLLECTION CudaException_ThreadSynchronize();

    return true;
}

} // namespace Heightmap
