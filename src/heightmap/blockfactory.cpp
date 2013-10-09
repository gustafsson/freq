#include "blockfactory.h"
#include "TaskTimer.h"
#include "referenceinfo.h"
#include "glblock.h"
#include "blockcacheinfo.h"
#include "blockquery.h"
#include "blockkernel.h"

#include "GlException.h"
#include "computationkernel.h"

#include <boost/foreach.hpp>

//#define TIME_BLOCKFACTORY
#define TIME_BLOCKFACTORY if(0)

//#define INFO_COLLECTION
#define INFO_COLLECTION if(0)

//#define VERBOSE_COLLECTION
#define VERBOSE_COLLECTION if(0)

// Limit the amount of memory used for caches by memoryUsedForCaches < freeMemory*MAX_FRACTION_FOR_CACHES
#define MAX_FRACTION_FOR_CACHES (1.f/2.f)

using namespace boost;
using namespace Signal;

namespace Heightmap {

BlockFactory::
        BlockFactory(BlockCache::Ptr cache, BlockLayout bl, VisualizationParams::ConstPtr vp, unsigned frame_counter)
    :
      cache_(cache),
      block_layout_(bl),
      visualization_params_(vp),
      _frame_counter(frame_counter),
      _free_memory(availableMemoryForSingleAllocation())
{
}


pBlock BlockFactory::
        createBlock( const Reference& ref )
{
    TIME_BLOCKFACTORY TaskTimer tt(format("New block %s") % ReferenceInfo(ref, block_layout_, visualization_params_));

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
        // Swallow silently and return null. Same reason as 'BlockFactory::attempt::catch (const CudaException& x)'.
        TaskInfo("BlockFactory::createBlock swallowed CudaException.\n%s", x.what());
        printCacheSize();
        return pBlock();
    }
#endif
    catch (const GlException& x )
    {
        // Swallow silently and return null. Same reason as 'BlockFactory::attempt::catch (const CudaException& x)'.
        TaskTimer("BlockFactory::createBlock swallowed GlException.\n%s", x.what()).suppressTiming();
        return pBlock();
    }

    EXCEPTION_ASSERT( result );

    // result is non-zero

    TIME_BLOCKFACTORY ComputationSynchronize();

    return result;
}


pBlock BlockFactory::
        attempt( const Reference& ref )
{
    try {
        INFO_COLLECTION TaskTimer tt("Allocation attempt");

        GlException_CHECK_ERROR();
        ComputationCheckError();

        pBlock attempt( new Block( ref, block_layout_, visualization_params_ ));
        Region r = RegionFactory( block_layout_ )( ref );
        EXCEPTION_ASSERT_LESS( r.a.scale, 1 );
        EXCEPTION_ASSERT_LESS_OR_EQUAL( r.b.scale, 1 );
        attempt->glblock.reset( new GlBlock( block_layout_, r.time(), r.scale() ));

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
        TaskInfo tt("BlockFactory::attempt swallowed CudaException.\n%s", x.what());
        printCacheSize();
    }
#endif
    catch (const GlException& x)
    {
        // Swallow silently and return null. Same reason as 'BlockFactory::attempt::catch (const CudaException& x)'.
        TaskInfo("BlockFactory::attempt swallowed GlException.\n%s", x.what());
    }
    return pBlock();
}



pBlock BlockFactory::
        getAllocatedBlock(const Reference& ref)
{    
    pBlock block;

    BlockCache::WritePtr cache(cache_);

    if (0!= "Reuse old redundant blocks")
    {
        // prefer to use block rather than discard an old block and then reallocate it

        // _recent is ordered with the most recently accessed blocks first,
        pBlock stealedBlock;
        if (!cache->recent ().empty())
        {
            stealedBlock = cache->recent ().back();

            // blocks that were used last frame has age 1 and blocks that were used this frame has age 0
            unsigned age = _frame_counter - stealedBlock->frame_number_last_used;

            if (1 < age)
            {
                size_t memForNewBlock = 0;
                memForNewBlock += sizeof(float); // OpenGL VBO
                memForNewBlock += sizeof(float); // Cuda device memory
                memForNewBlock += sizeof(float); // OpenGL texture
                memForNewBlock += sizeof(float); // OpenGL neareset texture
                memForNewBlock *= block_layout_.texels_per_block ();
                size_t allocatedMemory = BlockCacheInfo::cacheByteSize (cache->cache (), block_layout_);

                size_t margin = 2*memForNewBlock;

                if (allocatedMemory+memForNewBlock+margin > _free_memory*MAX_FRACTION_FOR_CACHES)
                {
                    pBlock stealedBlock = cache->recent ().back();

                    TaskInfo ti(format("Stealing block %s last used %u frames ago for %s. Total free %s, total cache %s, %u blocks")
                                % stealedBlock->reference ()
                                % (_frame_counter - stealedBlock->frame_number_last_used)
                                % ref
                                % DataStorageVoid::getMemorySizeText( _free_memory )
                                % DataStorageVoid::getMemorySizeText( allocatedMemory )
                                % cache->cache ().size()
                                 );

                    block.reset( new Block(ref, block_layout_, visualization_params_) );
                    block->glblock = stealedBlock->glblock;
                    write1(block->block_data())->cpu_copy.reset( new DataStorage<float>(block->glblock->heightSize()) );

                    stealedBlock->glblock.reset();
                    const Region& r = block->getRegion();
                    block->glblock->reset( r.time(), r.scale() );

                    // this doesn't belong here, if there's not enough memory
                    // available blockfactory should just fail rather than
                    // attempting a fix
                    cache->erase (stealedBlock->reference ());
                }

                // Need to release even more blocks? Release one at a time for each call to createBlock
                if (allocatedMemory > _free_memory*MAX_FRACTION_FOR_CACHES)
                {
                    pBlock back = cache->recent ().back();

                    size_t blockMemory = back->glblock->allocated_bytes_per_element()*block_layout_.texels_per_block ();
                    allocatedMemory -= std::min(allocatedMemory,blockMemory);
                    _free_memory = _free_memory > blockMemory ? _free_memory + blockMemory : 0;

                    TaskInfo(format("Soon removing block %s last used %u frames ago in favor of %s. Freeing %s, total free %s, cache %s, %u blocks")
                                 % back->reference ()
                                 % (_frame_counter - back->frame_number_last_used)
                                 % ref
                                 % DataStorageVoid::getMemorySizeText( blockMemory )
                                 % DataStorageVoid::getMemorySizeText( _free_memory )
                                 % DataStorageVoid::getMemorySizeText( allocatedMemory )
                                 % cache->cache ().size()
                                 );

                    cache->erase (stealedBlock->reference ());
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
        s*=block_layout_.texels_per_block ();
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

    bool empty_cache = cache->cache().empty();

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


void BlockFactory::
        setDummyValues( pBlock block )
{
    GlBlock::pHeight h = block->glblock->height();
    float* p = h->data->getCpuMemory();
    unsigned samples = block_layout_.texels_per_row (),
            scales = block_layout_.texels_per_column ();
    for (unsigned s = 0; s<samples/2; s++) {
        for (unsigned f = 0; f<scales; f++) {
            p[ f*samples + s] = 0.05f  +  0.05f * sin(s*10./samples) * cos(f*10./scales);
        }
    }
}


void BlockFactory::
        createBlockFromOthers( pBlock block )
{
    VERBOSE_COLLECTION TaskTimer tt("Stubbing new block");

    const Reference& ref = block->reference ();
    Intervals things_to_update = block->getInterval ();
    std::vector<pBlock> gib = BlockQuery(cache_).getIntersectingBlocks ( things_to_update.spannedInterval(), false, _frame_counter );

    const Region& r = block->getRegion ();

    int merge_levels = 10;

    VERBOSE_COLLECTION TaskTimer tt2("Checking %u blocks out of %u blocks, %d times", gib.size(), read1(cache_)->cache().size(), merge_levels);

    for (int merge_level=0; merge_level<merge_levels && things_to_update; ++merge_level)
    {
        VERBOSE_COLLECTION TaskTimer tt("%d, %s", merge_level, things_to_update.toString().c_str());

        std::vector<pBlock> next;
        BOOST_FOREACH ( const pBlock& bl, gib ) try
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

                    // Lock if available but don't wait for it to become available
                    BlockData::ReadPtr bd(bl->block_data (), 0);
                    mergeBlock( *block, *bl, *write1(block->block_data()), *bd );
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
        } catch (const BlockData::LockFailed&) {}

        gib = next;
    }
}


bool BlockFactory::
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
            VERBOSE_COLLECTION TaskInfo(boost::format("Using block %s") % ReferenceInfo(inBlock.reference (), block_layout_, visualization_params_ ));
        }
    }

    TIME_BLOCKFACTORY ComputationSynchronize();

    return true;
}


void BlockFactory::
        gc()
{
    BlockCache::WritePtr cache(cache_);
    const BlockCache::cache_t C = cache->cache (); // copy
    TaskTimer tt("Collection doing garbage collection", C.size());
    BlockCacheInfo::printCacheSize(C, block_layout_);
    TaskInfo("Of which %u are recently used", cache->recent().size());

    for (BlockCache::cache_t::const_iterator itr = C.begin(); itr!=C.end(); itr++ )
    {
        if (_frame_counter != itr->second->frame_number_last_used )
            cache->erase( itr->first );
    }
}


void BlockFactory::
        test()
{
    // It should create new blocks to make them ready for receiving transform data and rendering.
    {

    }
}


} // namespace Heightmap
