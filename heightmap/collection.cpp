#include "collection.h"
#include "block.cu.h"
#include "blockfilter.h"

#include "tfr/cwtfilter.h"
#include "tfr/cwt.h"
#include "signal/operationcache.h"

// Gpumisc
#include <InvokeOnDestruction.hpp>
#include <CudaException.h>
#include <GlException.h>
#include <string>
#include <neat_math.h>
#include <debugmacros.h>
#include <Statistics.h>
#include <cudaUtil.h>

// MSVC-GCC-compatibility workarounds
#include <msc_stdc.h>

#define TIME_COLLECTION
//#define TIME_COLLECTION if(0)

//#define INFO_COLLECTION
#define INFO_COLLECTION if(0)

//#define VERBOSE_COLLECTION
#define VERBOSE_COLLECTION if(0)

//#define VERBOSE_EACH_FRAME_COLLECTION
#define VERBOSE_EACH_FRAME_COLLECTION if(0)

//#define TIME_GETBLOCK
#define TIME_GETBLOCK if(0)

// Limit the amount of memory used for caches by memoryUsedForCaches < freeMemory*MAX_FRACTION_FOR_CACHES
#define MAX_FRACTION_FOR_CACHES (1.f/5.f)
#define MAX_CREATED_BLOCKS_PER_FRAME 8

using namespace Signal;

namespace Heightmap {


///// HEIGHTMAP::BLOCK

Block::
        ~Block()
{
    //TaskInfo("Deleting block %s", ref.toString().c_str());
}


///// HEIGHTMAP::COLLECTION

Collection::
        Collection( pOperation target )
:   target( target ),
    _samples_per_block( -1 ), // Created for each
    _scales_per_block( -1 ),
    _unfinished_count(0),
    _created_count(0),
    _frame_counter(0),
    _prev_length(.0f),
    _amplitude_axis(AmplitudeAxis_5thRoot)
{
    BOOST_ASSERT( target );
    samples_per_block( 1<<9 );
    scales_per_block( 1<<9 ); // sets _max_sample_size.scale

    TaskTimer tt("%s = %p", __FUNCTION__, this);

    _display_scale.setLinear(target->sample_rate());

    // set _max_sample_size.time
    invalidate_samples(Signal::Intervals::Intervals_ALL);
}


Collection::
        ~Collection()
{
    TaskInfo ti("%s = %p", __FUNCTION__, this);
    reset();
}


void Collection::
        reset()
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif
    INFO_COLLECTION
    {
        TaskInfo ti("Reset, cache count = %u, size = %g MB", _cache.size(), cacheByteSize()/1024.f/1024.f);
        foreach(const cache_t::value_type& b, _cache)
        {
            TaskInfo("%s", b.first.toString().c_str());
        }

        TaskInfo ti2("of which recent count %u", _recent.size());
        foreach(const recent_t::value_type& b, _recent)
        {
            TaskInfo("%s", b->ref.toString().c_str());
        }
    }
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
    reset();
#ifndef SAWE_NO_MUTEX
	QMutexLocker l(&_cache_mutex);
#endif
    _scales_per_block=v;
    _max_sample_size.scale = 1.f/_scales_per_block;
}


void Collection::
        samples_per_block(unsigned v)
{
    reset();
#ifndef SAWE_NO_MUTEX
	QMutexLocker l(&_cache_mutex);
#endif
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
    foreach(const recent_t::value_type& a, _recent)
    {
        Block const& b = *a;
        if (b.frame_number_last_used == _frame_counter)
        {
            const Reference &r = b.ref;
            // poke children
            poke(r.left());
            poke(r.right());
            poke(r.top());
            poke(r.bottom());

            // poke parent
            poke(r.parent());

            // poke surrounding sibblings
            Reference q = r;
            for (q.block_index[0] = std::max(1u, r.block_index[0]) - 1;
                 q.block_index[0] <= r.block_index[0] + 1;
                 q.block_index[0]++)
            {
                for (q.block_index[1] = std::max(1u, r.block_index[1]) - 1;
                     q.block_index[1] <= r.block_index[1] + 1;
                     q.block_index[1]++)
                {
                    poke(q);
                }
            }
        }
    }

    foreach(const cache_t::value_type& b, _cache)
    {
        b.second->glblock->unmap();

        if (b.second->frame_number_last_used != _frame_counter)
        {
            if (b.second->glblock->has_texture())
            {
                TaskTimer tt("Deleting texture");
                b.second->glblock->delete_texture();
            }
        }
    }

    _frame_counter++;

    return t;
}


void Collection::
        poke(const Reference& r)
{
    cache_t::iterator itr = _cache.find( r );
    if (itr != _cache.end())
        itr->second->frame_number_last_used = _frame_counter;
}


Signal::Intervals Collection::
        getInvalid(const Reference& r)
{
    cache_t::iterator itr = _cache.find( r );
    if (itr != _cache.end())
    {
        return r.getInterval() - itr->second->valid_samples;
    }

    return Signal::Intervals();
}


Reference Collection::
        entireHeightmap()
{
    Reference r(this);
    r.log2_samples_size = tvector<2,int>( floor_log2( _max_sample_size.time ), floor_log2( _max_sample_size.scale ));
    r.block_index = tvector<2,unsigned>(0,0);
    return r;
}


pBlock Collection::
        getBlock( const Reference& ref )
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
        else
        {
            TaskInfo("Delaying creation of block %s", ref.toString().c_str());
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
        if ((I & pb->ref.getInterval()).count())
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


Tfr::pTransform Collection::
        transform()
{
    Tfr::Filter* filter = dynamic_cast<Tfr::Filter*>(_filter.get());
    if (filter)
        return filter->transform();
    return Tfr::pTransform();
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
        discardOutside(Signal::Interval I)
{
    for (cache_t::iterator itr = _cache.begin(); itr!=_cache.end(); )
    {
        Signal::Interval blockInterval = itr->second->ref.getInterval();
        if ( 0 == (I & blockInterval).count() )
        {
            _recent.remove(itr->second);
            itr = _cache.erase(itr);
        } else {
            itr++;
        }
    }
}


void Collection::
        display_scale(Tfr::FreqAxis a)
{
    if (_display_scale == a)
        return;

    _display_scale = a;
    invalidate_samples( target->getInterval() );
}


void Collection::
        amplitude_axis(Heightmap::AmplitudeAxis a)
{
    if (_amplitude_axis == a)
        return;

    _amplitude_axis = a;
    invalidate_samples( target->getInterval() );
}


void Collection::
        invalidate_samples( const Intervals& sid )
{
    INFO_COLLECTION TaskTimer tt("Invalidating Heightmap::Collection, %s",
                                 sid.toString().c_str());

    float length = target->number_of_samples()/target->sample_rate();
    _max_sample_size.time = 2.f*std::max(1.f, length)/_samples_per_block;

#ifndef SAWE_NO_MUTEX
	QMutexLocker l(&_cache_mutex);
#endif
    foreach ( const cache_t::value_type& c, _cache )
		c.second->valid_samples -= sid;

    // If the signal has gotten shorter, make sure to discard all blocks that
    // go outside the new shorter interval
    if (_prev_length > length)
    {
        Interval I(target->number_of_samples(), Interval::IntervalType_MAX);
        for (cache_t::iterator itr = _cache.begin(); itr!=_cache.end(); )
        {
            Signal::Interval blockInterval = itr->second->ref.getInterval();
            if ( 0 < (I & blockInterval).count() )
            {
                _recent.remove(itr->second);
                itr = _cache.erase(itr);
            } else {
                itr++;
            }
        }
    }

    _prev_length = length;
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

    foreach ( const recent_t::value_type& a, _recent )
    {
        Block const& b = *a;
        if (_frame_counter == b.frame_number_last_used)
        {
            counter++;
            Intervals i = b.ref.getInterval();

            i -= b.valid_samples;

            r |= i;

            VERBOSE_EACH_FRAME_COLLECTION
            if (i)
                TaskInfo("block %s is invalid on %s", b.ref.toString().c_str(), i.toString().c_str());
        } else
            break;
    }
    }

    //TIME_COLLECTION TaskInfo("%u blocks with invalid samples %s", counter, r.toString().c_str());

    // If all recently used block are up-to-date then also update all their children, if any children are allocated
    if (!r)
    {
        foreach(const recent_t::value_type& a, _recent)
        {
            Block const& b = *a;
            if (b.frame_number_last_used == _frame_counter)
            {
                const Reference &p = b.ref;
                // children
                r |= getInvalid(p.left());
                r |= getInvalid(p.right());
                r |= getInvalid(p.top());
                r |= getInvalid(p.bottom());

                // parent
                r |= getInvalid(p.parent());

                // surrounding sibblings
                Reference q = p;
                for (q.block_index[0] = std::max(1u, p.block_index[0]) - 1;
                     q.block_index[0] <= p.block_index[0] + 1;
                     q.block_index[0]++)
                {
                    for (q.block_index[1] = std::max(1u, p.block_index[1]) - 1;
                         q.block_index[1] <= p.block_index[1] + 1;
                         q.block_index[1]++)
                    {
                        r |= getInvalid(q);
                    }
                }
            }
        }
    }
    return r;
}

////// private


pBlock Collection::
        attempt( const Reference& ref )
{
    try {
        INFO_COLLECTION TaskTimer tt("Allocation attempt");

        pBlock attempt( new Block(ref));
        Position a,b;
        ref.getArea(a,b);
        BOOST_ASSERT( a.scale < 1 && b.scale <= 1 );
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
        createBlock( const Reference& ref )
{
    pBlock result;
    INFO_COLLECTION TaskTimer tt("Creating a new block %s", ref.toString().c_str());
    // Try to create a new block
    try
    {
        pBlock block;

        if (0!= "Reuse old redundant blocks")
        {
            unsigned youngest_age = -1, youngest_count = 0;
            foreach ( const recent_t::value_type& b, _recent )
            {
                unsigned age = _frame_counter - b->frame_number_last_used;
                if (age < youngest_age) {
                    youngest_age = age;
                    youngest_count = 1;
                } else if (age <= youngest_age + 1) {
                    ++youngest_count;
                } else {
                    break; // _recent is ordered with the most recently accessed blocks first
                }
            }

            size_t memForNewBlock = 0;
            memForNewBlock += sizeof(float); // OpenGL VBO
            memForNewBlock += sizeof(float); // Cuda device memory
            memForNewBlock += 2*sizeof(float); // OpenGL texture, 2 times the size for mipmaps
            memForNewBlock += 2*sizeof(float2); // OpenGL texture, 2 times the size for mipmaps
            memForNewBlock *= scales_per_block()*samples_per_block();
            size_t allocatedMemory = this->cacheByteSize()*target->num_channels();
            size_t free = availableMemoryForSingleAllocation();

            size_t margin = memForNewBlock;
            // prefer to use block rather than discard an old block and then reallocate it
            if (youngest_count < _recent.size() && allocatedMemory+memForNewBlock+margin> free*MAX_FRACTION_FOR_CACHES)
            {
                block = _recent.back();
                Position a,b;
                block->ref.getArea(a,b);

                TaskTimer tt("Stealing block [%g:%g, %g:%g]. %u blocks, recent %u blocks.",
                             a.time, a.scale, b.time, b.scale, _cache.size(), _recent.size());

                _recent.remove( block );
                _cache.erase( block->ref );

                block->ref = ref;
                block->valid_samples.clear();
                ref.getArea(a,b);
                block->glblock->reset( b.time-a.time, b.scale-a.scale );
            }

            // Need to release even more blocks?
            while (youngest_count < _recent.size() && allocatedMemory+memForNewBlock> free*MAX_FRACTION_FOR_CACHES)
            {
                Position a,b;
                _recent.back()->ref.getArea(a,b);

                size_t blockMemory = _recent.back()->glblock->allocated_bytes_per_element()*scales_per_block()*samples_per_block();
                allocatedMemory -= std::min(allocatedMemory,blockMemory);

                TaskTimer tt("Removing block [%g:%g, %g:%g] (freeing %g KB). %u remaining blocks, recent %u blocks.",
                             a.time, a.scale, b.time, b.scale, blockMemory/1024.f/1024.f, _cache.size(), _recent.size());

                _cache.erase(_recent.back()->ref);
                _recent.pop_back();
            }
        }

        // estimate if there is enough memory available
        size_t s = 0;
        s += sizeof(float); // OpenGL VBO
        s += sizeof(float); // Cuda device memory
        s += 2*sizeof(float); // OpenGL texture, 2 times the size for mipmaps
        s += 2*sizeof(float2); // OpenGL texture, 2 times the size for mipmaps
        s*=scales_per_block()*samples_per_block();
        s*=1.5f; // 50% arbitrary extra

        s += 4<<20; // 4 MB to stub stft
        size_t free = availableMemoryForSingleAllocation();

        if (s>free)
        {
            TaskInfo("Require %g MB free memory for new block (including margins), only %g MB available", s/1024.f/1024.f, free/1024.f/1024.f);
            return pBlock();
        }

        if (!block) // couldn't reuse recent block, allocate a new one instead
            block = attempt( ref );

		bool empty_cache;
		{
#ifndef SAWE_NO_MUTEX
            QMutexLocker l(&_cache_mutex);
#endif
			empty_cache = _cache.empty();
		}

        if ( !block && !empty_cache) {
            TaskTimer tt("Memory allocation failed creating new block %s. Doing garbage collection", ref.toString().c_str());
            gc();
            block = attempt( ref ); // try again
        }
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_cache_mutex); // Keep in scope for the remainder of this function
#endif

        if ( 0 == block.get()) {
            TaskTimer tt("Failed creating new block %s", ref.toString().c_str());
            return pBlock(); // return null-pointer
        }

        // set to zero
        GlBlock::pHeight h = block->glblock->height();
        cudaMemset( h->data->getCudaGlobal().ptr(), 0, h->data->getSizeInBytes1D() );
        // will be unmapped in next_frame() or right before it's drawn

        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

        bool stubWithStft = true;
        BlockFilter* blockFilter = dynamic_cast<BlockFilter*>(_filter.get());
        if (blockFilter)
            stubWithStft = blockFilter->stubWithStft();

        VERBOSE_COLLECTION TaskTimer tt("Stubbing new block");

        Intervals things_to_update = ref.getInterval();
        if ( 1 /* create from others */ )
        {
            {
                if (1) {
                    {
#ifndef SAWE_NO_MUTEX
                        l.unlock();
#endif
                        std::vector<pBlock> gib = getIntersectingBlocks( things_to_update.coveredInterval(), false );
#ifndef SAWE_NO_MUTEX
                        l.relock();
#endif

                        Position a,b;
                        ref.getArea(a,b);
                        foreach( const pBlock& bl, gib )
                        {
                            // The switch from high resolution blocks to low resolution blocks (or
                            // the other way around) should be as invisible as possible. Therefor
                            // the contents should be stubbed from whatever contents the previous
                            // blocks already have, even if the contents are out-of-date.
                            //
                            // i.e. we use bl->ref.getInterval() here instead of bl->valid_samples.
                            Interval v = bl->ref.getInterval();

                            // Check if these samples are still considered for update
                            if ( (things_to_update & v ).count() <= 1)
                                continue;

                            int d = bl->ref.log2_samples_size[0];
                            d -= ref.log2_samples_size[0];
                            d += bl->ref.log2_samples_size[1];
                            d -= ref.log2_samples_size[1];

                            if (d>=-2 && d<=2)
                            {
                                Position a2,b2;
                                bl->ref.getArea(a2,b2);
                                if (a2.scale <= a.scale && b2.scale >= b.scale )
                                {
                                    // 'bl' covers all scales in 'block' (not necessarily all time samples though)
                                    things_to_update -= v;
                                    mergeBlock( block, bl, 0 );
                                }
                                else if (bl->ref.log2_samples_size[1] + 1 == ref.log2_samples_size[1])
                                {
                                    // mergeBlock makes sure that their scales overlap more or less
                                    mergeBlock( block, bl, 0 );
                                }
                                else
                                {
                                    // don't bother with things that doesn't match very well
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

        // fill block by STFT during the very first frames
        if (stubWithStft) // don't stubb if they will be filled by stft shortly hereafter
        if (things_to_update)
        if (10 > _frame_counter) {
            //size_t stub_size
            //if ((b.time - a.time)*fast_source( target)->sample_rate()*sizeof(float) )
            try
            {
                INFO_COLLECTION TaskTimer tt("stft %s", things_to_update.toString().c_str());

                fillBlock( block, things_to_update );
                things_to_update.clear();

                CudaException_CHECK_ERROR();
            }
            catch (const CudaException& x )
            {
                // Swallow silently, it is not fatal if this stubbed fft can't be computed right away
                TaskInfo tt("Collection::fillBlock swallowed CudaException.\n%s", x.what());
                printCacheSize();
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

    // result is non-zero
    _cache[ result->ref ] = result;

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
            if ((I - c->cached_samples()).empty())
            {
                return itr;
            }
        }

        if ( (I - itr->zeroed_samples_recursive()).empty() )
            return itr;

        pOperation s = itr->source();
        if (!s)
            return itr;
        
        itr = s;
    }
}


void Collection::
        fillBlock( pBlock block, const Signal::Intervals& to_update )
{
    StftToBlock stftmerger(this);
    Tfr::Stft* transp;
    stftmerger.transform( Tfr::pTransform( transp = new Tfr::Stft() ));
    transp->set_approximate_chunk_size(1 << 12); // 4096
    stftmerger.exclude_end_block = true;

    // Only take 1 MB of signal data at a time
    unsigned section_size = (1<<20) / sizeof(float);
    Intervals sections = to_update;
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

    INFO_COLLECTION TaskTimer tt("%s, %s into %s", __FUNCTION__,
                                 inBlock->ref.toString().c_str(), outBlock->ref.toString().c_str());

    GlBlock::pHeight out_h = outBlock->glblock->height();
    GlBlock::pHeight in_h = inBlock->glblock->height();

    BOOST_ASSERT( in_h.get() != out_h.get() );
    BOOST_ASSERT( outBlock.get() != inBlock.get() );

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
