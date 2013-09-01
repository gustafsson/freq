#include "collection.h"
#include "blockkernel.h"
#include "blockfilter.h"
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
using namespace boost;

namespace Heightmap {


Collection::
        Collection( TfrMapping tfr_mapping )
:   tfr_mapping_( tfr_mapping ),
    _is_visible( true ),
    _created_count(0),
    _frame_counter(0),
    _prev_length(.0f),
    _free_memory(availableMemoryForSingleAllocation()),
    failed_allocation_(false),
    failed_allocation_prev_(false)
{
    _max_sample_size.scale = 1.f/tfr_mapping_.block_size.texels_per_column ();

    // set _max_sample_size.time
    reset();
}


Collection::
        ~Collection()
{
    reset();
}


void Collection::
        reset()
{
    VERBOSE_COLLECTION {
        TaskInfo ti("Collection::Reset, cache count = %u, size = %s", _cache.size(), DataStorageVoid::getMemorySizeText( cacheByteSize() ).c_str() );
        ReferenceRegion rr(this->tfr_mapping ().block_size);
        BOOST_FOREACH (const cache_t::value_type& b, _cache)
        {
            TaskInfo(format("%s") % rr(b.first));
        }

        TaskInfo ti2("of which recent count %u", _recent.size());
        BOOST_FOREACH (const recent_t::value_type& b, _recent)
        {
            TaskInfo(format("%s") % rr(b->reference()));
        }
    }

    BOOST_FOREACH (const cache_t::value_type& v, _cache)
    {
        pBlock b = v.second;
        b->glblock.reset ();
    }

    BOOST_FOREACH (const pBlock b, _recent)
        b->glblock.reset ();

    BOOST_FOREACH (const pBlock b, _to_remove)
        b->glblock.reset ();

    _cache.clear();
    _recent.clear();

/*
    invalidate_samples(Signal::Intervals::Intervals_ALL);
*/
}


bool Collection::
        empty()
{
    return _cache.empty() && _recent.empty();
}


void Collection::
        next_frame()
{
    VERBOSE_EACH_FRAME_COLLECTION TaskTimer tt(boost::format("%s(), %u, %u")
            % __FUNCTION__ % _created_count % _recent.size());


    _created_count = 0;

    ReferenceRegion rr(this->tfr_mapping ().block_size);

    recent_t delayremoval;
    BOOST_FOREACH(const recent_t::value_type& b, _to_remove)
    {
        if (b.unique ())
            TaskInfo(format("Release block %s") % rr (b->reference()));
        else
        {
            delayremoval.push_back (b);
        }
    }
    if (!delayremoval.empty ())
        TaskInfo(format("Delaying removal of %d blocks") % delayremoval.size ());

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
            VERBOSE_COLLECTION TaskTimer tt("Deleting texture");
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
}


void Collection::
        poke(pBlock b)
{
/*
    TaskInfo ti(boost::format("poke %s") % b->getRegion ());
    TaskInfo(boost::format("b->frame_number_last_used = %1%") % b->frame_number_last_used);
    TaskInfo(boost::format("_frame_counter = %1%") % _frame_counter);
    TaskInfo(boost::format("b->getInterval() = %s") % b->getInterval());
    TaskInfo(boost::format("b->valid_samples = %s") % b->valid_samples);
    TaskInfo(boost::format("recently_created_ = %s") % recently_created_);
*/

    b->to_delete = false;
    b->frame_number_last_used = _frame_counter;
    _recent.remove( b );
    _recent.push_front( b );
}


Reference Collection::
        entireHeightmap() const
{
    Reference r;
    r.log2_samples_size = Reference::Scale( floor_log2( _max_sample_size.time ), floor_log2( _max_sample_size.scale ));
    r.block_index = Reference::Index(0,0);
    return r;
}


pBlock Collection::
        getBlock( const Reference& ref )
{
    // Look among cached blocks for this reference
    TIME_GETBLOCK TaskTimer tt(format("getBlock %s") % ReferenceInfo(ref, tfr_mapping ()));

    pBlock block = findBlock( ref );

    if (!block)
    {
        if ( MAX_CREATED_BLOCKS_PER_FRAME > _created_count )
        {
            block = createBlock( ref );
            _created_count++;
            recently_created_ |= block->getInterval ();
        }
        else
        {
            failed_allocation_ = true;
            TaskInfo(format("Delaying creation of block %s") % ReferenceRegion(tfr_mapping().block_size)(ref));
        }
    }

    if (block)
    {
        BlockData::ReadPtr bd(block->block_data ());

        if (block->new_data_available) {
            *block->glblock->height()->data = *bd->cpu_copy; // 256 KB memcpy < 100 us (256*256*4 = 256 KB, about 52 us)
            block->new_data_available = false;
        }

        poke(block);
    }

    return block;
}


pBlock Collection::
        findBlock( const Reference& ref )
{
    cache_t::iterator itr = _cache.find( ref );
    if (itr != _cache.end())
        return itr->second;

    return pBlock();
}


std::vector<pBlock> Collection::
        getIntersectingBlocks( const Intervals& I, bool only_visible ) const
{
    std::vector<pBlock> r;
    r.reserve(32);

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


unsigned long Collection::
        cacheByteSize() const
{
    // For each block there may be both a slope map and heightmap. Also there
    // may be both a texture and a vbo, and possibly a mapped cuda copy.
    //
    // But most of the blocks will only have a heightmap vbo, and none of the
    // others.

    unsigned long sumsize = 0;

    BOOST_FOREACH (const cache_t::value_type& b, _cache)
        {
        sumsize += b.second->glblock->allocated_bytes_per_element();
        }

    BOOST_FOREACH (const pBlock& b, _to_remove)
        {
        unsigned abpe = b->glblock->allocated_bytes_per_element();
        if (abpe > sumsize)
            sumsize = 0;
        else
            sumsize -= abpe;
        }

    unsigned elements_per_block = tfr_mapping_.block_size.texels_per_block ();
    return sumsize*elements_per_block;
}


unsigned Collection::
        cacheCount() const
{
    return _cache.size();
}


void Collection::
        printCacheSize() const
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
    for (cache_t::iterator itr = _cache.begin(); itr!=_cache.end(); ++itr)
    {
        pBlock block(itr->second);
        Signal::Interval blockInterval = ReferenceInfo(itr->first, tfr_mapping_).getInterval();
        Signal::Interval toKeep = I & blockInterval;
        bool remove_entire_block = toKeep == Signal::Interval();
        bool keep_entire_block = toKeep == blockInterval;
        if ( remove_entire_block )
        {
            removeBlock(block);
        }
        else if ( keep_entire_block )
        {
        }
        else
        {
            // clear partial block
            if( I.first <= blockInterval.first && I.last < blockInterval.last )
            {
                Region ir = ReferenceRegion(tfr_mapping_)(itr->first);
                float t = I.last / tfr_mapping_.targetSampleRate - ir.a.time;

                BlockData::WritePtr bd(block->block_data());

                ReferenceInfo ri(itr->first, tfr_mapping_);
                ::blockClearPart( bd->cpu_copy,
                              ceil(t * ri.sample_rate()) );

                block->new_data_available = true;
            }
        }
    }
}


bool Collection::
        failed_allocation()
{
    bool r = failed_allocation_ || failed_allocation_prev_;
    failed_allocation_prev_ = failed_allocation_;
    failed_allocation_ = false;
    return r;
}


bool Collection::
        isVisible() const
{
    return _is_visible;
}


void Collection::
        setVisible(bool v)
{
    _is_visible = v;
}


const TfrMapping& Collection::
        tfr_mapping() const
{
    return tfr_mapping_;
}


void Collection::
        tfr_mapping(TfrMapping new_tfr_mapping)
{
    float length = new_tfr_mapping.length;

    // If only the length has changed, don't invalidate the entire heightmap.
    TfrMapping tfr_mapping_length = tfr_mapping_;
    tfr_mapping_length.length = new_tfr_mapping.length;
    bool doreset = !(new_tfr_mapping == tfr_mapping_length);

    tfr_mapping_ = new_tfr_mapping;
    _max_sample_size.scale = 1.f/tfr_mapping_.block_size.texels_per_column ();
    _max_sample_size.time = 2.f*std::max(1.f, length)/tfr_mapping_.block_size.texels_per_row ();

    // If the signal has gotten shorter, make sure to discard all blocks that
    // go outside the new shorter interval
    if (_prev_length > length)
        discardOutside( Signal::Interval(0, length*tfr_mapping_.targetSampleRate) );

    _prev_length = length;

    if (doreset)
        reset();
//    else
//        invalidate_samples( Signal::Interval::Interval_ALL );
}


Intervals Collection::
        needed_samples(UnsignedIntervalType& smallest_length)
{
    Intervals r;

    if (!_is_visible)
        return r;

    BOOST_FOREACH ( const recent_t::value_type& a, _recent )
    {
        Block& b = *a;
        unsigned framediff = _frame_counter - b.frame_number_last_used;
        if (1 == framediff || 0 == framediff) // this block was used last frame or this frame
        {
            Interval i = b.getInterval();
            r |= i;
            if (i.count () < smallest_length)
                smallest_length = i.count ();
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

        pBlock attempt( new Block( ref, tfr_mapping_ ));
        Region r = ReferenceRegion( tfr_mapping_ )( ref );
        EXCEPTION_ASSERT( r.a.scale < 1 && r.b.scale <= 1 );
        attempt->glblock.reset( new GlBlock( tfr_mapping ().block_size, r.time(), r.scale() ));

        write1(attempt->block_data())->cpu_copy.reset( new DataStorage<float>(attempt->glblock->heightSize()) );

/*
        {
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

Signal::Intervals Collection::
        recently_created()
{
    Signal::Intervals r = recently_created_;
    recently_created_.clear ();
    return r;
}


pBlock Collection::
        createBlock( const Reference& ref )
{
    TIME_COLLECTION TaskTimer tt(format("New block %s") % ReferenceInfo(ref, tfr_mapping ()));

    pBlock result;
    // Try to create a new block
    try
    {
        pBlock block = getAllocatedBlock(ref);
        if (!block)
            return block;

        // set to zero

        // DataStorage makes sure nothing actually happens here unless
        // cpu_copy has already been allocated (i.e if it is stolen)
        // Each block is about 1 MB so this takes about 0.5-1 ms.
        write1(block->block_data())->cpu_copy->ClearContents ();
        block->new_data_available = true;

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
    _cache[ result->reference() ] = result;

    TIME_COLLECTION ComputationSynchronize();

    return result;
}


void Collection::
        removeBlock (pBlock b)
{
    _recent.remove(b);
    _cache.erase(b->reference());
    _to_remove.push_back( b );
    b->to_delete = true;
    b->glblock.reset ();
}


pBlock Collection::
        getAllocatedBlock(const Reference& ref)
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
                memForNewBlock *= tfr_mapping_.block_size.texels_per_block ();
                size_t allocatedMemory = this->cacheByteSize();

                size_t margin = 2*memForNewBlock;

                if (allocatedMemory+memForNewBlock+margin > _free_memory*MAX_FRACTION_FOR_CACHES)
                {
                    pBlock stealedBlock = _recent.back();

                    TaskInfo ti(format("Stealing block %s last used %u frames ago for %s. Total free %s, total cache %s, %u blocks")
                                % stealedBlock->reference ()
                                % (_frame_counter - stealedBlock->frame_number_last_used)
                                % ref
                                % DataStorageVoid::getMemorySizeText( _free_memory )
                                % DataStorageVoid::getMemorySizeText( allocatedMemory )
                                % _cache.size()
                                 );

                    block.reset( new Block(ref, tfr_mapping_) );
                    block->glblock = stealedBlock->glblock;
                    write1(block->block_data())->cpu_copy.reset( new DataStorage<float>(block->glblock->heightSize()) );

                    stealedBlock->glblock.reset();
                    const Region& r = block->getRegion();
                    block->glblock->reset( r.time(), r.scale() );

                    removeBlock(stealedBlock);
                }

                // Need to release even more blocks? Release one at a time for each call to createBlock
                if (allocatedMemory > _free_memory*MAX_FRACTION_FOR_CACHES)
                {
                    pBlock back = _recent.back();

                    size_t blockMemory = back->glblock->allocated_bytes_per_element()*tfr_mapping_.block_size.texels_per_block ();
                    allocatedMemory -= std::min(allocatedMemory,blockMemory);
                    _free_memory = _free_memory > blockMemory ? _free_memory + blockMemory : 0;

                    TaskInfo(format("Soon removing block %s last used %u frames ago in favor of %s. Freeing %s, total free %s, cache %s, %u blocks")
                                 % back->reference ()
                                 % (_frame_counter - back->frame_number_last_used)
                                 % ref
                                 % DataStorageVoid::getMemorySizeText( blockMemory )
                                 % DataStorageVoid::getMemorySizeText( _free_memory )
                                 % DataStorageVoid::getMemorySizeText( allocatedMemory )
                                 % _cache.size()
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
        s*=tfr_mapping_.block_size.texels_per_block ();
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
        TaskTimer tt(format("Memory allocation failed creating new block %s. Doing garbage collection") % ref);
        gc();
        block = attempt( ref ); // try again
    }

    if ( !block ) {
        TaskTimer tt(format("Failed to create a new block %s") % ref);
        return pBlock(); // return null-pointer
    }

    return block;
}


void Collection::
        setDummyValues( pBlock block )
{
    GlBlock::pHeight h = block->glblock->height();
    float* p = h->data->getCpuMemory();
    unsigned samples = tfr_mapping_.block_size.texels_per_row (),
            scales = tfr_mapping_.block_size.texels_per_column ();
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
            if ( (things_to_update & v).count() <= 1)
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
                    mergeBlock( *block, *bl, *write1(block->block_data()), *read1(bl->block_data()) );
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
        mergeBlock( Block& outBlock, const Block& inBlock, BlockData& outData, const BlockData& inData )
{
    EXCEPTION_ASSERT( &outBlock != &inBlock );

    // Find out what intervals that match
    Intervals transferDesc = inBlock.getInterval() & outBlock.getInterval();

    // If the blocks doesn't overlap, there's nothing to do
    if (!transferDesc)
        return false;

    const Region& ri = inBlock.getRegion();
    const Region& ro = outBlock.getRegion();

    // If the blocks doesn't overlap, there's nothing to do
    if (ro.b.scale <= ri.a.scale || ri.b.scale<=ro.a.scale || ro.b.time <= ri.a.time || ri.b.time <= ro.a.time)
        return false;

    INFO_COLLECTION TaskTimer tt(boost::format("%s, %s into %s") % __FUNCTION__ % ri % ro);

    INFO_COLLECTION ComputationSynchronize();

    // TODO examine why blockMerge is really really slow
    {
        INFO_COLLECTION TaskTimer tt("blockMerge");
        ::blockMerge( inData.cpu_copy,
                      outData.cpu_copy,
                      ResampleArea( ri.a.time, ri.a.scale, ri.b.time, ri.b.scale ),
                      ResampleArea( ro.a.time, ro.a.scale, ro.b.time, ro.b.scale ) );
        INFO_COLLECTION ComputationSynchronize();
    }

    outBlock.new_data_available = true;

    //bool isCwt = dynamic_cast<const Tfr::Cwt*>(transform());
    //bool using_subtexel_aggregation = !isCwt || (renderer ? renderer->redundancy()<=1 : false);
    bool using_subtexel_aggregation = false;

#ifndef CWT_SUBTEXEL_AGGREGATION
    // subtexel aggregation is way to slow
    using_subtexel_aggregation = false;
#endif

    if (!using_subtexel_aggregation) // blockMerge doesn't support subtexel aggregation
        if (false) // disabled validation of blocks until this is stable
    {
        // Validate region of block if inBlock was source of higher resolution than outBlock
        if (inBlock.reference().log2_samples_size[0] < outBlock.reference().log2_samples_size[0] &&
            inBlock.reference().log2_samples_size[1] == outBlock.reference().log2_samples_size[1] &&
            ri.b.scale==ro.b.scale && ri.a.scale==ro.a.scale)
        {
            VERBOSE_COLLECTION TaskInfo(boost::format("Using block %s") % ReferenceInfo(inBlock.reference (), tfr_mapping ()));
        }
    }

    TIME_COLLECTION ComputationSynchronize();

    return true;
}


} // namespace Heightmap
