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
#include <boost/foreach.hpp>
#include <boost/unordered_set.hpp>

// std
#include <string>

// MSVC-GCC-compatibility workarounds
#include <msc_stdc.h>


//#define TIME_COLLECTION
#define TIME_COLLECTION if(0)

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
    block_configuration_( this ),
    _unfinished_count(0),
    _created_count(0),
    _frame_counter(0),
    _prev_length(.0f),
    _free_memory(availableMemoryForSingleAllocation())
{
    EXCEPTION_ASSERT( target );
    samples_per_block( 1<<9 );
    scales_per_block( 1<<9 ); // sets _max_sample_size.scale

    TaskTimer tt("%s = %p", __FUNCTION__, this);

    Tfr::FreqAxis fa;
    fa.setLinear(target->sample_rate());
    display_scale(fa);

    // set _max_sample_size.time
    reset();

    block_config_.reset (new BlockConfiguration(this));
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
        BOOST_FOREACH (const cache_t::value_type& b, _cache)
        {
            TaskInfo("%s", b.first.toString().c_str());
        }

        TaskInfo ti2("of which recent count %u", _recent.size());
        BOOST_FOREACH (const recent_t::value_type& b, _recent)
        {
            TaskInfo("%s", b->reference().toString().c_str());
        }
    }

    _cache.clear();
    _recent.clear();

#ifndef SAWE_NO_MUTEX
    l.unlock();
#endif

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


unsigned Collection::
        scales_per_block() const
{
    return block_configuration ().scalesPerBlock();
}


unsigned Collection::
        samples_per_block() const
{
    return block_configuration ().samplesPerBlock();
}


void Collection::
        scales_per_block(unsigned v)
{
    BlockConfiguration bc = block_configuration();
    bc.scalesPerBlock ( v );
    block_configuration ( bc );
}


void Collection::
        samples_per_block(unsigned v)
{
    BlockConfiguration bc = block_configuration();
    bc.samplesPerBlock (v);
    block_configuration ( bc );
}


unsigned Collection::
        next_frame()
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif

    boost::shared_ptr<TaskTimer> tt;
    if (_unfinished_count)
        VERBOSE_EACH_FRAME_COLLECTION tt.reset( new TaskTimer("%s(), %u, %u, %u",
            __FUNCTION__, _unfinished_count, _created_count, (unsigned)_recent.size()) );


    unsigned t = _unfinished_count;
    _unfinished_count = 0;
    _created_count = 0;


    recent_t delayremoval;
    BOOST_FOREACH(const recent_t::value_type& b, _to_remove)
    {
        _recent.remove(b);
        _cache.erase(b->reference());

        if (b.unique ())
            TaskInfo("Release block %s", b->reference().toString().c_str());
        else
        {
            TaskInfo("Delaying removal of block %s", b->reference().toString().c_str());
            delayremoval.push_back (b);
        }
    }
    _to_remove = delayremoval;


    boost::unordered_set<Reference> blocksToPoke;

    BOOST_FOREACH(const cache_t::value_type& b, _cache)
    {
        Block* block = b.second.get();
        if (block->frame_number_last_used == _frame_counter)
        {
            // Mark these blocks and surrounding blocks as in-use
            blocksToPoke.insert (block->reference ());
        }
        else if (block->glblock->has_texture ())
        {
            // This block isn't used but it has allocated a texture in OpenGL
            // memory that can easily recreate as soon as it is needed.
            TaskTimer tt("Deleting texture");
            block->glblock->delete_texture ();
        }
    }

    boost::unordered_set<Reference> blocksToPoke2 = blocksToPoke;

    BOOST_FOREACH(const Reference& r, blocksToPoke)
    {
        // poke blocks that are likely to be needed soon

        // poke children
        blocksToPoke2.insert (r.left());
        blocksToPoke2.insert (r.right());
        blocksToPoke2.insert (r.top());
        blocksToPoke2.insert (r.bottom());

        // poke parent
        blocksToPoke2.insert (r.parent());
        blocksToPoke2.insert (r.parentHorizontal());
        blocksToPoke2.insert (r.parentVertical());

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
                blocksToPoke2.insert (q);
            }
        }
    }


    BOOST_FOREACH(const Reference& r, blocksToPoke2)
    {
        cache_t::iterator itr = _cache.find( r );
        if (itr != _cache.end())
            poke(itr->second);
    }

    _free_memory = availableMemoryForSingleAllocation();

    _frame_counter++;

    return t;
}


void Collection::
        poke(pBlock b)
{
#ifndef SAWE_NO_MUTEX
    b->to_delete = false;
#endif
    b->frame_number_last_used = _frame_counter;
    _recent.remove( b );
    _recent.push_front( b );
}


Signal::Intervals inline Collection::
        getInvalid(const Reference& r) const
{
    cache_t::const_iterator itr = _cache.find( r );
    if (itr != _cache.end())
    {
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&itr->second->cpu_copy_mutex);
#endif
        return itr->second->getInterval() - itr->second->valid_samples;
    }

    return Signal::Intervals();
}


Reference Collection::
        entireHeightmap()
{
    Reference r( this->block_configuration () );
    r.log2_samples_size = tvector<2,int>( floor_log2( _max_sample_size.time ), floor_log2( _max_sample_size.scale ));
    r.block_index = tvector<2,unsigned>(0,0);
    return r;
}


pBlock Collection::
        getBlock( const Reference& ref )
{
    // Look among cached blocks for this reference
    TIME_GETBLOCK TaskTimer tt("getBlock %s", ref.toString().c_str());

    pBlock block = findBlock( ref );

    if (!block)
    {
        if ( MAX_CREATED_BLOCKS_PER_FRAME > _created_count )
        {
            block = createBlock( ref );
            _created_count++;
        }
        else
        {
            TaskInfo("Delaying creation of block %s", ref.toString().c_str());
        }
    }

    if (block)
    {
        #ifndef SAWE_NO_MUTEX
            if (block->new_data_available) {
                QMutexLocker l(&block->cpu_copy_mutex);
                *block->glblock->height()->data = *block->cpu_copy; // 256 KB memcpy < 100 us (256*256*4 = 256 KB, about 52 us)
                block->new_data_available = false;
            }
        #endif

        Intervals blockInt = block->getInterval();
        if (blockInt -= block->valid_samples)
            _unfinished_count++;

        #ifndef SAWE_NO_MUTEX
            QMutexLocker l(&_cache_mutex);
        #endif

        poke(block);
    }

    return block;
}


pBlock Collection::
        findBlock( const Reference& ref )
{
    #ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
    #endif

    cache_t::iterator itr = _cache.find( ref );
    if (itr != _cache.end())
        return itr->second;

    return pBlock();
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

    BOOST_FOREACH( const cache_t::value_type& c, _cache )
    {
        const pBlock& pb = c.second;
        
        unsigned framediff = _frame_counter - pb->frame_number_last_used;
        if (only_visible && framediff != 0 && framediff != 1)
            continue;

#ifndef SAWE_NO_MUTEX
        if (pb->to_delete)
            continue;
#endif

        if ((I & pb->getInterval()).count())
            r.push_back(pb);
    }

/*
    // consistency check
    foreach( const cache_t::value_type& c, _cache )
    {
        bool found = false;
        foreach( const recent_t::value_type& r, _recent )
        {
            if (r == c.second)
                found = true;
        }

        EXCEPTION_ASSERT(found);
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

        EXCEPTION_ASSERT(found);
    }
*/

    return r;
}


const Tfr::TransformDesc* Collection::
        transform()
{
    Tfr::Filter* filter = dynamic_cast<Tfr::Filter*>(_filter.get());
    if (filter)
        return filter->transform()->transformDesc();
    return 0;
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
    BOOST_FOREACH (const cache_t::value_type& b, _cache)
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

    for (cache_t::iterator itr = _cache.begin(); itr!=_cache.end(); itr++ )
    {
        if (_frame_counter != itr->second->frame_number_last_used )
            removeBlock( itr->second );
    }
}


void Collection::
        discardOutside(Signal::Interval I)
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif

    for (cache_t::iterator itr = _cache.begin(); itr!=_cache.end(); ++itr)
    {
        Signal::Interval blockInterval = itr->first.getInterval();
        Signal::Interval toKeep = I & blockInterval;
        if ( !toKeep )
            removeBlock(itr->second);
        else if ( blockInterval == toKeep )
        {
        }
        else
        {
            if( I.first <= blockInterval.first && I.last < blockInterval.last )
            {
                bool hasValidOutside = itr->second->non_zero & ~Signal::Intervals(I);
                if (hasValidOutside)
                {
                    Region ir = ReferenceInfo(block_config_.get (), itr->first).getRegion();
                    float t = I.last / target->sample_rate() - ir.a.time;

#ifdef SAWE_NO_MUTEX
                    GlBlock::pHeight block = itr->second->glblock->height();
                    Block::pData data = block->data;
#else
                    QMutexLocker l(&itr->second->cpu_copy_mutex);
                    Block::pData data = itr->second->cpu_copy;
                    itr->second->new_data_available = true;
#endif

                    ::blockClearPart( data,
                                  ceil(t * itr->first.sample_rate()) );

                    itr->second->valid_samples &= I;
                    itr->second->non_zero &= I;
                }
            }
        }
    }
}


void Collection::
        display_scale(Tfr::FreqAxis a)
{
    BlockConfiguration bc = block_configuration();
    bc.display_scale ( a );
    block_configuration( bc );
}


void Collection::
        amplitude_axis(Heightmap::AmplitudeAxis a)
{
    BlockConfiguration bc = block_configuration();
    bc.amplitude_axis ( a );
    block_configuration( bc );
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


const BlockConfiguration& Collection::
        block_configuration() const
{
    return block_configuration_;
}


void Collection::
        block_configuration(BlockConfiguration& new_block_config)
{
    bool doreset = false;

    {
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_cache_mutex);
#endif
        doreset =
                new_block_config.scalesPerBlock () != block_configuration_.scalesPerBlock () ||
                new_block_config.samplesPerBlock () != block_configuration_.samplesPerBlock ();

        block_configuration_ = new_block_config;

        _max_sample_size.scale = 1.f/block_configuration_.scalesPerBlock ();
    }

    if (doreset)
        reset();
    else
        invalidate_samples( Signal::Interval::Interval_ALL );
}


void Collection::
        invalidate_samples( const Intervals& sid )
{
    INFO_COLLECTION TaskTimer tt("Invalidating Heightmap::Collection, %s",
                                 sid.toString().c_str());

    {
#ifndef SAWE_NO_MUTEX
        QMutexLocker l(&_cache_mutex);
#endif
        BOOST_FOREACH ( const cache_t::value_type& c, _cache )
            c.second->valid_samples -= sid;
    }

    // validate length
    Interval wholeSignal = target->getInterval();
    float length = wholeSignal.last / target->sample_rate();
    _max_sample_size.time = 2.f*std::max(1.f, length)/samples_per_block();

    // If the signal has gotten shorter, make sure to discard all blocks that
    // go outside the new shorter interval
    if (_prev_length > length)
        discardOutside( wholeSignal );

    _prev_length = length;
}


Intervals Collection::
        invalid_samples()
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

    BOOST_FOREACH ( const recent_t::value_type& a, _recent )
    {
        Block& b = *a;
        unsigned framediff = _frame_counter - b.frame_number_last_used;
        if (1 == framediff || 0 == framediff) // this block was used last frame or this frame
        {
#ifndef SAWE_NO_MUTEX
            QMutexLocker l(&b.cpu_copy_mutex);
#endif

            counter++;
            Intervals i = b.getInterval();

            i -= b.valid_samples;

            r |= i;

            VERBOSE_EACH_FRAME_COLLECTION
            {
                bool to_delete = false;
#ifndef SAWE_NO_MUTEX
                to_delete = b.to_delete;
#endif
                if (i)
                    TaskInfo("block %s is invalid on %s%s", b.reference().toString().c_str(), i.toString().c_str(), to_delete?" to delete":"");
                else
                    TaskInfo("block %s is valid%s", b.reference().toString().c_str(), to_delete?" to delete":"");
            }
        } else
            break;
    }
    }

    TIME_COLLECTION TaskInfo("%u recently used blocks. Total invalid samples %s", counter, r.toString().c_str());

    // If all recently used block are up-to-date then also update all their children, if any children are allocated
    if (false) if (!r)
    {
        //TIME_COLLECTION TaskTimer tt("Collection::invalid_samples recent_t, %u, %p", _recent.size(), this);
        BOOST_FOREACH (const recent_t::value_type& a, _recent)
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

        ReferenceInfo refinfo( block_config_.get (), ref );
        pBlock attempt( new Block(refinfo));
        Region r = refinfo.getRegion();
        EXCEPTION_ASSERT( r.a.scale < 1 && r.b.scale <= 1 );
        attempt->glblock.reset( new GlBlock( this, r.time(), r.scale() ));

#ifndef SAWE_NO_MUTEX
        attempt->cpu_copy.reset( new DataStorage<float>(attempt->glblock->heightSize()) );
#endif
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
    TIME_COLLECTION TaskTimer tt("Creating a new block %s", ref.toString().c_str());

    pBlock result;
    // Try to create a new block
    try
    {
        pBlock block = getAllocatedBlock(ref);
        if (!block)
            return block;

        // set to zero
#ifdef SAWE_NO_MUTEX
        GlBlock::pHeight h = block->glblock->height();
        h->data->ClearContents();
        // will be unmapped right before it's drawn
#else
        QMutexLocker cpul(&block->cpu_copy_mutex);
        // DataStorage makes sure nothing actually happens here unless
        // cpu_copy has already been allocated (i.e if it is stolen)
        // Each block is about 1 MB so this takes about 500-1000 us.
        block->cpu_copy->ClearContents ();
#endif

        createBlockFromOthers( block );

        // For some filters a block could be created with valid content from existing blocks
        //BlockFilter* blockFilter = dynamic_cast<BlockFilter*>(_filter.get());
        //if (!blockFilter->createFromOthers())
        //    block->valid_samples.clear();

        //setDummyValues(block);

        result = block;
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

    EXCEPTION_ASSERT( result );

    // result is non-zero
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif
    _cache[ result->reference() ] = result;

    TIME_COLLECTION ComputationSynchronize();

    return result;
}


void Collection::
        removeBlock (pBlock b)
{
    _to_remove.push_back( b );
#ifndef SAWE_NO_MUTEX
    b->to_delete = true;
#endif
}


pBlock Collection::
        getAllocatedBlock(const Reference& ref)
{
#ifndef SAWE_NO_MUTEX
    QMutexLocker l(&_cache_mutex);
#endif

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
    #ifndef SAWE_NO_MUTEX
                    QMutexLocker l(&stealedBlock->cpu_copy_mutex);
    #endif
                    pBlock stealedBlock = _recent.back();

                    TaskInfo ti("Stealing block %s last used %u frames ago for %s. Total free %s, total cache %s, %u blocks",
                                stealedBlock->reference ().toString ().c_str (),
                                 _frame_counter - stealedBlock->frame_number_last_used,
                                ref.toString ().c_str (),
                                 DataStorageVoid::getMemorySizeText( _free_memory ).c_str(),
                                 DataStorageVoid::getMemorySizeText( allocatedMemory ).c_str(),
                                  _cache.size()
                                 );

                    _recent.remove( stealedBlock );
                    _cache.erase( stealedBlock->reference() );

                    ReferenceInfo refinfo( this->block_config_.get (), ref );
                    block.reset( new Block(refinfo) );
                    block->glblock = stealedBlock->glblock;
    #ifndef SAWE_NO_MUTEX
                    block->cpu_copy = stealedBlock->cpu_copy;
    #endif
                    stealedBlock->glblock.reset();
                    const Region& r = block->getRegion();
                    block->glblock->reset( r.time(), r.scale() );
                }

                // Need to release even more blocks? Release one at a time for each call to createBlock
                if (allocatedMemory > _free_memory*MAX_FRACTION_FOR_CACHES)
                {
                    pBlock back = _recent.back();

                    size_t blockMemory = back->glblock->allocated_bytes_per_element()*scales_per_block()*samples_per_block();
                    allocatedMemory -= std::min(allocatedMemory,blockMemory);
                    _free_memory = _free_memory > blockMemory ? _free_memory + blockMemory : 0;

                    TaskInfo("Soon removing block %s last used %u frames ago in favor of %s. Freeing %s, total free %s, cache %s, %u blocks",
                                 back->reference ().toString().c_str (),
                                 _frame_counter - block->frame_number_last_used,
                                 ref.toString ().c_str (),
                                 DataStorageVoid::getMemorySizeText( blockMemory ).c_str(),
                                 DataStorageVoid::getMemorySizeText( _free_memory ).c_str(),
                                 DataStorageVoid::getMemorySizeText( allocatedMemory ).c_str(),
                                 _cache.size()
                                 );

                    removeBlock(back);
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

        if (s>_free_memory)
        {
            TaskInfo("Require %s free memory for new block (including margins), only %s available",
                     DataStorageVoid::getMemorySizeText( s ).c_str(),
                     DataStorageVoid::getMemorySizeText( _free_memory ).c_str());
            return pBlock();
        }

        block = attempt( ref );
    }

    bool empty_cache = _cache.empty();

    if ( !block && !empty_cache ) {
        TaskTimer tt("Memory allocation failed creating new block %s. Doing garbage collection", ref.toString().c_str());
        gc();
        block = attempt( ref ); // try again
    }

    if ( !block ) {
        TaskTimer tt("Failed to create a new block %s", ref.toString().c_str());
        return pBlock(); // return null-pointer
    }

    return block;
}


void Collection::
        setDummyValues( pBlock block )
{
    GlBlock::pHeight h = block->glblock->height();
    float* p = h->data->getCpuMemory();
    unsigned samples = samples_per_block (),
            scales = scales_per_block ();
    for (unsigned s = 0; s<samples/2; s++) {
        for (unsigned f = 0; f<scales; f++) {
            p[ f*samples + s] = 0.05f  +  0.05f * sin(s*10./samples) * cos(f*10./scales);
        }
    }
}


void Collection::
        createBlockFromOthers( pBlock block )
{
    VERBOSE_COLLECTION TaskTimer tt("Stubbing new block");

    const Reference& ref = block->reference ();
    Intervals things_to_update = block->getInterval ();
    std::vector<pBlock> gib = getIntersectingBlocks ( things_to_update.spannedInterval(), false );

    const Region& r = block->getRegion ();

    int merge_levels = 10;
    #ifndef SAWE_NO_MUTEX
    // Don't bother locking _cache to check size during debugging
    #endif
    VERBOSE_COLLECTION TaskTimer tt2("Checking %u blocks out of %u blocks, %d times", gib.size(), _cache.size(), merge_levels);

    for (int merge_level=0; merge_level<merge_levels && things_to_update; ++merge_level)
    {
        VERBOSE_COLLECTION TaskTimer tt("%d, %s", merge_level, things_to_update.toString().c_str());

        std::vector<pBlock> next;
        BOOST_FOREACH ( const pBlock& bl, gib )
        {
            // The switch from high resolution blocks to low resolution blocks (or
            // the other way around) should be as invisible as possible. Therefor
            // the contents should be stubbed from whatever contents the previous
            // blocks already have, even if the contents are out-of-date.
            //
            Interval v = bl->getInterval ();

            // Check if these samples are still considered for update
            if ( (things_to_update & v & bl->non_zero).count() <= 1)
                continue;

            int d = bl->reference().log2_samples_size[0];
            d -= ref.log2_samples_size[0];
            d += bl->reference().log2_samples_size[1];
            d -= ref.log2_samples_size[1];

            if (d==merge_level || -d==merge_level)
            {
                const Region& r2 = bl->getRegion ();
                if (r2.a.scale <= r.a.scale && r2.b.scale >= r.b.scale )
                {
                    // 'bl' covers all scales in 'block' (not necessarily all time samples though)
                    things_to_update -= v;
                    mergeBlock( block, bl, 0 );
                }
                else if (bl->reference ().log2_samples_size[1] + 1 == ref.log2_samples_size[1])
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
                next.push_back ( bl );
            }
        }
        gib = next;
    }
}


bool Collection::
        mergeBlock( pBlock outBlock, pBlock inBlock, unsigned /*cuda_stream*/ )
{
#ifndef SAWE_NO_MUTEX
    EXCEPTION_ASSERT( outBlock.get() != inBlock.get() );
    EXCEPTION_ASSERT( !outBlock->cpu_copy_mutex.tryLock() );
    QMutexLocker il (&inBlock->cpu_copy_mutex);
#endif

    // Find out what intervals that match
    Intervals transferDesc = inBlock->getInterval() & inBlock->valid_samples & outBlock->getInterval();

    // Remove already computed intervals
    transferDesc -= outBlock->valid_samples;

    // If block is already up to date, abort merge
    if (transferDesc.empty())
        return false;

    const Region& ri = inBlock->getRegion();
    const Region& ro = outBlock->getRegion();

    if (ro.b.scale <= ri.a.scale || ri.b.scale<=ro.a.scale || ro.b.time <= ri.a.time || ri.b.time <= ro.a.time)
        return false;

    INFO_COLLECTION TaskTimer tt(boost::format("%s, %s into %s") % __FUNCTION__ % ri % ro);

#ifdef SAWE_NO_MUTEX
    GlBlock::pHeight out_h = outBlock->glblock->height();
    GlBlock::pHeight in_h = inBlock->glblock->height();
    EXCEPTION_ASSERT( in_h.get() != out_h.get() );

    Block::pData out_data = out_h->data;
    Block::pData in_data = in_h->data;
#else
    Block::pData out_data = outBlock->cpu_copy;
    Block::pData in_data = inBlock->cpu_copy;
    outBlock->new_data_available = true;
#endif
    EXCEPTION_ASSERT( in_data.get() != out_data.get() );

    INFO_COLLECTION ComputationSynchronize();

    // TODO examine why blockMerge is really really slow
    {
        INFO_COLLECTION TaskTimer tt("blockMerge");
        ::blockMerge( in_data,
                      out_data,
                      ResampleArea( ri.a.time, ri.a.scale, ri.b.time, ri.b.scale ),
                      ResampleArea( ro.a.time, ro.a.scale, ro.b.time, ro.b.scale ) );
        INFO_COLLECTION ComputationSynchronize();
    }

    outBlock->valid_samples -= inBlock->getInterval();

    bool isCwt = dynamic_cast<const Tfr::Cwt*>(transform());
    bool using_subtexel_aggregation = !isCwt || (renderer ? renderer->redundancy()<=1 : false);

#ifndef CWT_SUBTEXEL_AGGREGATION
    // subtexel aggregation is way to slow
    using_subtexel_aggregation = false;
#endif

    if (!using_subtexel_aggregation) // blockMerge doesn't support subtexel aggregation
        if (false) // disabled validation of blocks until this is stable
    {
        // Validate region of block if inBlock was source of higher resolution than outBlock
        if (inBlock->reference().log2_samples_size[0] < outBlock->reference().log2_samples_size[0] &&
            inBlock->reference().log2_samples_size[1] == outBlock->reference().log2_samples_size[1] &&
            ri.b.scale==ro.b.scale && ri.a.scale==ro.a.scale)
        {
            outBlock->valid_samples -= inBlock->getInterval();
            outBlock->valid_samples |= inBlock->valid_samples & inBlock->getInterval() & outBlock->getInterval ();
            VERBOSE_COLLECTION TaskTimer tt("Using block %s", (inBlock->valid_samples & outBlock->getInterval ()).toString().c_str() );
        }
    }

    TIME_COLLECTION ComputationSynchronize();

    return true;
}


} // namespace Heightmap
