#include "collection.h"
#include "blockkernel.h"
#include "blockfilter.h"
#include "renderer.h"
#include "glblock.h"

#include "tfr/cwt.h"
#include "tfr/stft.h"
#include "signal/operationcache.h"

// Gpumisc
#include <GlException.h>
#include <neat_math.h>
#include <computationkernel.h>

// boost
#include <boost/date_time/posix_time/posix_time.hpp>

// std
#include <string>

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
#define MAX_FRACTION_FOR_CACHES (1.f/2.f)
#define MAX_CREATED_BLOCKS_PER_FRAME 16

using namespace Signal;

namespace Heightmap {


Collection::
        Collection( pOperation target )
:   target( target ),
    renderer( 0 ),
    _is_visible( true ),
    _samples_per_block( -1 ), // Created for each
    _scales_per_block( -1 ),
    _unfinished_count(0),
    _created_count(0),
    _frame_counter(0),
    _prev_length(.0f),
    _free_memory(availableMemoryForSingleAllocation()),
    _amplitude_axis(AmplitudeAxis_5thRoot)
{
    BOOST_ASSERT( target );
    samples_per_block( 1<<9 );
    scales_per_block( 1<<9 ); // sets _max_sample_size.scale

    TaskTimer tt("%s = %p", __FUNCTION__, this);

    _display_scale.setLinear(target->sample_rate());

    // set _max_sample_size.time
    reset();
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

    {
        TaskInfo ti("Collection::Reset, cache count = %u, size = %s", _cache.size(), DataStorageVoid::getMemorySizeText( cacheByteSize() ).c_str() );
        foreach(const cache_t::value_type& b, _cache)
        {
            TaskInfo("%s", b.first.toString().c_str());
        }

        TaskInfo ti2("of which recent count %u", _recent.size());
        foreach(const recent_t::value_type& b, _recent)
        {
            TaskInfo("%s", b->reference().toString().c_str());
        }
    }

    _cache.clear();
    _recent.clear();

    invalidate_samples(Signal::Intervals::Intervals_ALL);
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
    {
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_cache_mutex);
#endif
        _scales_per_block=v;
        _max_sample_size.scale = 1.f/_scales_per_block;
    }

    reset();
}


void Collection::
        samples_per_block(unsigned v)
{
    {
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_cache_mutex);
#endif
        _samples_per_block=v;
    }

    reset();
}


unsigned Collection::
        next_frame()
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif

    // TaskTimer tt("%s, _recent.size() = %lu", __FUNCTION__, _recent.size());

    boost::shared_ptr<TaskTimer> tt;
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

    std::vector<pBlock> blocksToPoke;

    foreach(const cache_t::value_type& b, _cache)
    {
        b.second->glblock->unmap();

        if (b.second->frame_number_last_used == _frame_counter)
        {
            blocksToPoke.push_back(b.second);
        }
        else
        {
            if (b.second->glblock->has_texture())
            {
                TaskTimer tt("Deleting texture");
                b.second->glblock->delete_texture();
            }
        }
    }


    foreach(const pBlock& b, blocksToPoke)
    {
        // poke blocks that are likely to be needed soon

        const Reference &r = b->reference();
        // poke children
        poke(r.left());
        poke(r.right());
        poke(r.top());
        poke(r.bottom());

        // poke parent
        poke(r.parent());

        // poke surrounding sibblings, and this
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

    _free_memory = availableMemoryForSingleAllocation();

    _frame_counter++;

    return t;
}


void Collection::
        poke(const Reference& r)
{
    cache_t::iterator itr = _cache.find( r );
    if (itr != _cache.end())
    {
        itr->second->frame_number_last_used = _frame_counter;
        _recent.remove( itr->second );
        _recent.push_front( itr->second );
    }
}


Signal::Intervals inline Collection::
        getInvalid(const Reference& r) const
{
    cache_t::const_iterator itr = _cache.find( r );
    if (itr != _cache.end())
    {
        return itr->second->getInterval() - itr->second->valid_samples;
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
        Intervals refInt = block->getInterval();
        if (!(refInt-=block->valid_samples).empty())
            _unfinished_count++;

		#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_cache_mutex);
		#endif

        block->frame_number_last_used = _frame_counter;
        _recent.remove( block );
        _recent.push_front( block );
    }

    return block;
}

std::vector<pBlock> Collection::
        getIntersectingBlocks( const Intervals& I, bool only_visible )
{
    std::vector<pBlock> r;
    r.reserve(32);

	#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
	#endif
    //TIME_COLLECTION TaskTimer tt("getIntersectingBlocks( %s, %s ) from %u caches", I.toString().c_str(), only_visible?"only visible":"all", _cache.size());

    //float fs = target->sample_rate();
    //float Ia = I.first / fs, Ib = I.last/fs;

    foreach ( const cache_t::value_type& c, _cache )
    {
        const pBlock& pb = c.second;
        
        unsigned framediff = _frame_counter - pb->frame_number_last_used;
        if (only_visible && framediff != 0 && framediff != 1)
            continue;

//        Position a, b;
//        // getArea is faster than getInterval but misses the overlapping parts returned by getInterval
//        pb->ref.getArea(a, b, _samples_per_block, _scales_per_block);
//        if (b.time <= Ia || a.time >= Ib)
//            continue;

        // This check is done in mergeBlock as well, but do it here first
        // for a hopefully more local and thus faster loop.
        if ((I & pb->getInterval()).count())
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
#ifdef USE_CUDA
    cudaMemGetInfo(&free, &total);
#endif
    TaskInfo("Currently has %u cached blocks (ca %s). There are %s graphics memory free of a total of %s",
             _cache.size(),
             DataStorageVoid::getMemorySizeText( cacheByteSize() ).c_str(),
             DataStorageVoid::getMemorySizeText( free ).c_str(),
             DataStorageVoid::getMemorySizeText( total ).c_str());
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
            const Region& r = itr->second->getRegion();
            TaskTimer tt("Release block [%g, %g]", r.a.time, r.b.time);

            _recent.remove(itr->second);
            itr = _cache.erase(itr);
        } else {
            itr++;
        }
    }

    TaskInfo("Now has %u cached blocks, %s",
             _cache.size(),
             DataStorageVoid::getMemorySizeText(
                     _cache.size() * scales_per_block()*samples_per_block()*(1+2)*sizeof(float) ).c_str()
             );
}


void Collection::
        discardOutside(Signal::Interval I)
{
    for (cache_t::iterator itr = _cache.begin(); itr!=_cache.end(); )
    {
        Signal::Interval blockInterval = itr->first.getInterval();
        if ( 0 == (I & blockInterval).count() )
        {
            _recent.remove(itr->second);
            itr = _cache.erase(itr);
            continue;
        }
        else if ( blockInterval == (I & blockInterval))
        {
        }
        else
        {
            if( I.first <= blockInterval.first && I.last < blockInterval.last )
            {
                bool hasValidOutside = itr->second->non_zero & ~Signal::Intervals(I);
                if (hasValidOutside)
                {
                    Region ir = itr->first.getRegion();
                    float t = I.last / target->sample_rate() - ir.a.time;

                    GlBlock::pHeight block = itr->second->glblock->height();

                    ::blockClearPart( block->data,
                                  ceil(t * itr->first.sample_rate()) );

                    itr->second->valid_samples &= I;
                    itr->second->non_zero &= I;
                }
            }
        }
        itr++;
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


bool Collection::
        isVisible()
{
    return _is_visible;
}


void Collection::
        setVisible(bool v)
{
    _is_visible = v;
}


void Collection::
        invalidate_samples( const Intervals& sid )
{
    INFO_COLLECTION TaskTimer tt("Invalidating Heightmap::Collection, %s",
                                 sid.toString().c_str());

#ifndef SAWE_NO_MUTEX
	QMutexLocker l(&_cache_mutex);
#endif
    foreach ( const cache_t::value_type& c, _cache )
		c.second->valid_samples -= sid;

    // validate length
    Interval wholeSignal = target->getInterval();
    float length = wholeSignal.last / target->sample_rate();
    _max_sample_size.time = 2.f*std::max(1.f, length)/_samples_per_block;

    // If the signal has gotten shorter, make sure to discard all blocks that
    // go outside the new shorter interval
    if (_prev_length > length)
    {
        for (cache_t::iterator itr = _cache.begin(); itr!=_cache.end(); )
        {
            Signal::Interval blockInterval = itr->second->getInterval();
            if ( !(blockInterval & wholeSignal) )
            {
                _recent.remove(itr->second);
                itr = _cache.erase(itr);
            } else {
                itr->second->valid_samples &= wholeSignal;
                itr++;
            }
        }
    }

    _prev_length = length;
}


Intervals Collection::
        invalid_samples() const
{
    Intervals r;

    if (!_is_visible)
        return r;

#ifndef SAWE_NO_MUTEX
	QMutexLocker l(&_cache_mutex);
#endif

    unsigned counter = 0;
    {
    //TIME_COLLECTION TaskTimer tt("Collection::invalid_samples, %u, %p", _recent.size(), this);

    foreach ( const recent_t::value_type& a, _recent )
    {
        Block const& b = *a;
        unsigned framediff = _frame_counter - b.frame_number_last_used;
        if (1 == framediff || 0 == framediff) // this block was used last frame or this frame
        {
            counter++;
            Intervals i = b.getInterval();

            i -= b.valid_samples;

            r |= i;

            VERBOSE_EACH_FRAME_COLLECTION
            if (i)
                TaskInfo("block %s is invalid on %s", b.reference().toString().c_str(), i.toString().c_str());
        } else
            break;
    }
    }

    //TIME_COLLECTION TaskInfo("%u blocks with invalid samples %s", counter, r.toString().c_str());

    // If all recently used block are up-to-date then also update all their children, if any children are allocated
    if (false) if (!r)
    {
        //TIME_COLLECTION TaskTimer tt("Collection::invalid_samples recent_t, %u, %p", _recent.size(), this);
        foreach(const recent_t::value_type& a, _recent)
        {
            Block const& b = *a;
            if (b.frame_number_last_used == _frame_counter)
            {
                const Reference &p = b.reference();
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

        GlException_CHECK_ERROR();
        ComputationCheckError();

        pBlock attempt( new Block(ref));
        Region r = ref.getRegion();
        BOOST_ASSERT( r.a.scale < 1 && r.b.scale <= 1 );
        attempt->glblock.reset( new GlBlock( this, r.time(), r.scale() ));
/*        {
            GlBlock::pHeight h = attempt->glblock->height();
            //GlBlock::pSlope sl = attempt->glblock->slope();
        }
        attempt->glblock->unmap();
*/
        GlException_CHECK_ERROR();
        ComputationCheckError();

        return attempt;
    }
#ifdef USE_CUDA
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
#endif
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
            // prefer to use block rather than discard an old block and then reallocate it

            // _recent is ordered with the most recently accessed blocks first,
            pBlock stealedBlock;
            if (!_recent.empty())
            {
                stealedBlock = _recent.back();

                // blocks that were used last frame has age 1 and blocks that were used this frame has age 0
                unsigned age = _frame_counter - stealedBlock->frame_number_last_used;

                if (1 < age)
                {
                    size_t memForNewBlock = 0;
                    memForNewBlock += sizeof(float); // OpenGL VBO
                    memForNewBlock += sizeof(float); // Cuda device memory
                    memForNewBlock += sizeof(float); // OpenGL texture
                    memForNewBlock += sizeof(float); // OpenGL neareset texture
                    memForNewBlock *= scales_per_block()*samples_per_block();
                    size_t allocatedMemory = this->cacheByteSize()*target->num_channels();

                    size_t margin = 2*memForNewBlock;

                    if (allocatedMemory+memForNewBlock+margin > _free_memory*MAX_FRACTION_FOR_CACHES)
                    {
                        pBlock stealedBlock = _recent.back();
                        const Region& stealed_r = stealedBlock->getRegion();

                        TaskInfo("Stealing block [%g:%g, %g:%g] last used %u frames ago. Total free %s, total cache %s, %u blocks",
                                     stealed_r.a.time, stealed_r.b.time, stealed_r.a.scale, stealed_r.b.scale,
                                     _frame_counter - stealedBlock->frame_number_last_used,
                                     DataStorageVoid::getMemorySizeText( _free_memory ).c_str(),
                                     DataStorageVoid::getMemorySizeText( allocatedMemory ).c_str(),
                                      _cache.size()
                                     );

                        _recent.remove( stealedBlock );
                        _cache.erase( stealedBlock->reference() );

                        block.reset( new Block(ref) );
                        block->glblock = stealedBlock->glblock;
                        stealedBlock->glblock.reset();
                        const Region& r = block->getRegion();
                        block->glblock->reset( r.time(), r.scale() );
                    }

                    // Need to release even more blocks? Release one at a time for each call to createBlock
                    if (allocatedMemory > _free_memory*MAX_FRACTION_FOR_CACHES)
                    {
                        const Region& r = _recent.back()->getRegion();

                        size_t blockMemory = _recent.back()->glblock->allocated_bytes_per_element()*scales_per_block()*samples_per_block();
                        allocatedMemory -= std::min(allocatedMemory,blockMemory);
                        _free_memory = _free_memory > blockMemory ? _free_memory + blockMemory : 0;

                        VERBOSE_COLLECTION TaskInfo("Removing block [%g:%g, %g:%g] last used %u frames ago. Freeing %s, total free %s, cache %s, %u blocks",
                                     r.a.time, r.b.time, r.a.scale, r.b.scale,
                                     _frame_counter - block->frame_number_last_used,
                                     DataStorageVoid::getMemorySizeText( blockMemory ).c_str(),
                                     DataStorageVoid::getMemorySizeText( _free_memory ).c_str(),
                                     DataStorageVoid::getMemorySizeText( allocatedMemory ).c_str(),
                                     _cache.size()
                                     );

                        _cache.erase(_recent.back()->reference());
                        _recent.pop_back();
                    }
                }
            }
        }

        if (!block) // couldn't reuse recent block, allocate a new one instead
        {
            // estimate if there is enough memory available
            size_t s = 0;
            s += sizeof(float); // OpenGL VBO
            s += sizeof(float); // Cuda device memory
            s += 2*sizeof(float); // OpenGL texture, 2 times the size for mipmaps
            s += 2*sizeof(std::complex<float>); // OpenGL texture, 2 times the size for mipmaps
            s*=scales_per_block()*samples_per_block();
            s*=1.5f; // 50% arbitrary extra

            s += 4<<20; // 4 MB to stub stft

            if (s>_free_memory)
            {
                TaskInfo("Require %s free memory for new block (including margins), only %s available",
                         DataStorageVoid::getMemorySizeText( s ).c_str(),
                         DataStorageVoid::getMemorySizeText( _free_memory ).c_str());
                return pBlock();
            }

            block = attempt( ref );
        }

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

        VERBOSE_COLLECTION ComputationSynchronize();

        {
            VERBOSE_COLLECTION TaskTimer tt("ClearContents");
            // set to zero
            GlBlock::pHeight h = block->glblock->height();
            h->data->ClearContents();
            // will be unmapped in next_frame() or right before it's drawn

            VERBOSE_COLLECTION ComputationSynchronize();
        }

        GlException_CHECK_ERROR();
        ComputationCheckError();

        bool stubWithStft = true;
        BlockFilter* blockFilter = dynamic_cast<BlockFilter*>(_filter.get());
        if (blockFilter)
            stubWithStft = blockFilter->stubWithStft();

        VERBOSE_COLLECTION TaskTimer tt("Stubbing new block");

        Intervals things_to_update = block->getInterval();

        if ( 1 /* create from others */ )
        {
#ifndef SAWE_NO_MUTEX
            l.unlock();
#endif
            std::vector<pBlock> gib = getIntersectingBlocks( things_to_update.spannedInterval(), false );
#ifndef SAWE_NO_MUTEX
            l.relock();
#endif

            const Region& r = block->getRegion();

            int merge_levels = 10;
            VERBOSE_COLLECTION TaskTimer tt("Checking %u blocks out of %u blocks, %d times", gib.size(), _cache.size(), merge_levels);

            for (int merge_level=0; merge_level<merge_levels; ++merge_level)
            {
                VERBOSE_COLLECTION TaskTimer tt("%d, %s", merge_level, things_to_update.toString().c_str());

                std::vector<pBlock> next;
                foreach( const pBlock& bl, gib )
                {
                    // The switch from high resolution blocks to low resolution blocks (or
                    // the other way around) should be as invisible as possible. Therefor
                    // the contents should be stubbed from whatever contents the previous
                    // blocks already have, even if the contents are out-of-date.
                    //
                    // i.e. we use bl->ref.getInterval() here instead of bl->valid_samples.
                    Interval v = bl->getInterval();

                    // Check if these samples are still considered for update
                    if ( (things_to_update & v & bl->non_zero).count() <= 1)
                        continue;

                    int d = bl->reference().log2_samples_size[0];
                    d -= ref.log2_samples_size[0];
                    d += bl->reference().log2_samples_size[1];
                    d -= ref.log2_samples_size[1];

                    if (d==merge_level || -d==merge_level)
                    {
                        const Region& r2 = bl->getRegion();
                        if (r2.a.scale <= r.a.scale && r2.b.scale >= r.b.scale )
                        {
                            // 'bl' covers all scales in 'block' (not necessarily all time samples though)
                            things_to_update -= v;
                            mergeBlock( block, bl, 0 );
                        }
                        else if (bl->reference().log2_samples_size[1] + 1 == ref.log2_samples_size[1])
                        {
                            // mergeBlock makes sure that their scales overlap more or less
                            //mergeBlock( block, bl, 0 ); takes time
                        }
                        else
                        {
                            // don't bother with things that doesn't match very well
                        }
                    }
                    else
                    {
                        next.push_back( bl );
                    }
                }
                gib = next;
            }
        }


        if (!blockFilter->createFromOthers())
            block->valid_samples.clear();


        // fill block by STFT during the very first frames
        if (stubWithStft) // don't stubb if they will be filled by stft shortly hereafter
        if (things_to_update)
        if (10 > _frame_counter) {
            //size_t stub_size
            //if ((b.time - a.time)*fast_source( target)->sample_rate()*sizeof(float) )
#ifdef USE_CUDA
            try
            {
#endif
                INFO_COLLECTION TaskTimer tt("stft %s", things_to_update.toString().c_str());

                fillBlock( block, things_to_update );
                things_to_update.clear();

                ComputationCheckError();
#ifdef USE_CUDA
            }
            catch (const CudaException& x )
            {
                // Swallow silently, it is not fatal if this stubbed fft can't be computed right away
                TaskInfo tt("Collection::fillBlock swallowed CudaException.\n%s", x.what());
                printCacheSize();
            }
#endif
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
        ComputationCheckError();
    }
#ifdef USE_CUDA
    catch (const CudaException& x )
    {
        // Swallow silently and return null. Same reason as 'Collection::attempt::catch (const CudaException& x)'.
        TaskInfo("Collection::createBlock swallowed CudaException.\n%s", x.what());
        printCacheSize();
        return pBlock();
    }
#endif
    catch (const GlException& x )
    {
        // Swallow silently and return null. Same reason as 'Collection::attempt::catch (const CudaException& x)'.
        TaskTimer("Collection::createBlock swallowed GlException.\n%s", x.what()).suppressTiming();
        return pBlock();
    }

    BOOST_ASSERT( 0 != result.get() );

    // result is non-zero
    _cache[ result->reference() ] = result;

    TIME_COLLECTION ComputationSynchronize();

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
    const Interval outInterval = outBlock->getInterval();

    // Find out what intervals that match
    Intervals transferDesc = inBlock->getInterval();
    transferDesc &= outInterval;

    // Remove already computed intervals
    transferDesc -= outBlock->valid_samples;

    // If block is already up to date, abort merge
    if (transferDesc.empty())
        return false;

    const Region& ri = inBlock->getRegion();
    const Region& ro = outBlock->getRegion();

    if (ro.b.scale <= ri.a.scale || ri.b.scale<=ro.a.scale || ro.b.time <= ri.a.time || ri.b.time <= ro.a.time)
        return false;

    INFO_COLLECTION TaskTimer tt("%s, %s into %s", __FUNCTION__,
                                 inBlock->reference().toString().c_str(), outBlock->reference().toString().c_str());

    GlBlock::pHeight out_h = outBlock->glblock->height();
    GlBlock::pHeight in_h = inBlock->glblock->height();

    BOOST_ASSERT( in_h.get() != out_h.get() );
    BOOST_ASSERT( outBlock.get() != inBlock.get() );

    INFO_COLLECTION ComputationSynchronize();

    {
    INFO_COLLECTION TaskTimer tt("blockMerge");
    ::blockMerge( in_h->data,
                  out_h->data,

                  ResampleArea( ri.a.time, ri.a.scale, ri.b.time, ri.b.scale ),
                  ResampleArea( ro.a.time, ro.a.scale, ro.b.time, ro.b.scale ) );
    INFO_COLLECTION ComputationSynchronize();
    }

    outBlock->valid_samples -= inBlock->getInterval();

    bool isCwt = dynamic_cast<Tfr::Cwt*>(transform().get());
    bool using_subtexel_aggregation = !isCwt || (renderer ? renderer->redundancy()<=1 : false);

#ifndef CWT_SUBTEXEL_AGGREGATION
    // subtexel aggregation is way to slow
    using_subtexel_aggregation = false;
#endif

    if (!using_subtexel_aggregation) // blockMerge doesn't support subtexel aggregation
    {
        // Validate region of block if inBlock was source of higher resolution than outBlock
        if (inBlock->reference().log2_samples_size[0] < outBlock->reference().log2_samples_size[0] &&
            inBlock->reference().log2_samples_size[1] == outBlock->reference().log2_samples_size[1] &&
            ri.b.scale==ro.b.scale && ri.a.scale==ro.a.scale)
        {
            outBlock->valid_samples -= inBlock->getInterval();
            outBlock->valid_samples |= inBlock->valid_samples;
            outBlock->valid_samples &= outInterval;
            VERBOSE_COLLECTION TaskTimer tt("Using block %s", (inBlock->valid_samples & outInterval).toString().c_str() );
        }
    }

    TIME_COLLECTION ComputationSynchronize();

    return true;
}

} // namespace Heightmap
