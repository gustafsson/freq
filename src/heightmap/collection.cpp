#include "collection.h"
#include "glblock.h"
#include "blockfactory.h"
#include "blockquery.h"
#include "blockcacheinfo.h"
#include "tfr/cwt.h"
#include "tfr/stft.h"
#include "reference_hash.h"
#include "blocks/garbagecollector.h"
#include "blocks/merger.h"
#include "blocks/mergertexture.h"
#include "blocks/clearinterval.h"
#include "blocks/chunkmerger.h"

// Gpumisc
//#include "GlException.h"
#include "neat_math.h"
#include "computationkernel.h"
#include "tasktimer.h"

// boost
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/foreach.hpp>
#include <boost/unordered_set.hpp>

// std
#include <string>

// MSVC-GCC-compatibility workarounds
#include "msc_stdc.h"


//#define INFO_COLLECTION
#define INFO_COLLECTION if(0)

//#define VERBOSE_COLLECTION
#define VERBOSE_COLLECTION if(0)

//#define VERBOSE_EACH_FRAME_COLLECTION
#define VERBOSE_EACH_FRAME_COLLECTION if(0)

//#define TIME_GETBLOCK
#define TIME_GETBLOCK if(0)

#define MAX_CREATED_BLOCKS_PER_FRAME 16

using namespace Signal;
using namespace boost;

namespace Heightmap {


Collection::
        Collection( BlockLayout block_layout, VisualizationParams::ConstPtr visualization_params)
:   block_layout_( 2, 2, FLT_MAX ),
    visualization_params_(),
    cache_( new BlockCache ),
    merger_(),
    _is_visible( true ),
    _created_count(0),
    _frame_counter(0),
    _prev_length(.0f),
    failed_allocation_(false),
    failed_allocation_prev_(false)
{
    // set _max_sample_size
    this->block_layout (block_layout);
    this->visualization_params (visualization_params);
}


Collection::
        ~Collection()
{
    TaskInfo ti("~Collection");
    clear();
}


void Collection::
        clear()
{
    BlockCache::WritePtr cache(cache_);
    const BlockCache::cache_t& C = cache->cache ();
    VERBOSE_COLLECTION {
        TaskInfo ti("Collection::Reset, cache count = %u, size = %s", C.size(), DataStorageVoid::getMemorySizeText( BlockCacheInfo::cacheByteSize (C) ).c_str() );
        RegionFactory rr(block_layout_);
        BOOST_FOREACH (const BlockCache::cache_t::value_type& b, C)
        {
            TaskInfo(format("%s") % rr(b.first));
        }

        TaskInfo ti2("of which recent count %u", cache->recent().size());
        BOOST_FOREACH (const BlockCache::recent_t::value_type& b, cache->recent())
        {
            TaskInfo(format("%s") % rr(b->reference()));
        }
    }

    BOOST_FOREACH (const BlockCache::cache_t::value_type& v, C)
    {
        pBlock b = v.second;
        if (b->glblock && !b->glblock.unique ()) TaskInfo(boost::format("Error. glblock %s is in use %d") % b->getRegion () % b->glblock.use_count ());
        b->glblock.reset ();
    }

    BOOST_FOREACH (const pBlock b, cache->recent())
    {
        if (b->glblock && !b->glblock.unique ()) TaskInfo(boost::format("Error. glblock %s is in use %d") % b->getRegion () % b->glblock.use_count ());
        b->glblock.reset ();
    }

    BOOST_FOREACH (const pBlock b, _to_remove)
    {
        if (b->glblock && !b->glblock.unique ()) TaskInfo(boost::format("Error. glblock %s is in use %d") % b->getRegion () % b->glblock.use_count ());
        b->glblock.reset ();
    }

    cache->clear();
}


void Collection::
        next_frame()
{
    BlockCache::WritePtr cache(cache_);

    VERBOSE_EACH_FRAME_COLLECTION TaskTimer tt(boost::format("%s(), %u, %u")
            % __FUNCTION__ % _created_count % cache->recent().size());

    _created_count = 0;

    RegionFactory rr(block_layout_);

    BlockCache::recent_t delayremoval;
    BOOST_FOREACH(const toremove_t::value_type& b, _to_remove)
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

    BOOST_FOREACH(const BlockCache::cache_t::value_type& b, cache->cache())
    {
        Block* block = b.second.get();
        if (block->frame_number_last_used == _frame_counter)
        {
            // Mark these blocks and surrounding blocks as in-use
            blocksToPoke.insert (block->reference ());
        }
        else if (block->glblock->has_texture ())
        {
            unsigned framediff = _frame_counter - block->frame_number_last_used;
            if (framediff < 2) {
                // This can be verified easily without knowing the acutal frame number
                block->frame_number_last_used = 0;
            }

            // This block isn't used but it has allocated a texture in OpenGL
            // memory that can easily recreate as soon as it is needed.
//            VERBOSE_COLLECTION TaskTimer tt(boost::format("Deleting texture for block %s") % block->getRegion ());
            //block->glblock->delete_texture ();
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
        pBlock b = cache->probe (r);
        if (b)
            poke(b);
    }

    _frame_counter++;
}


unsigned Collection::
        frame_number() const
{
    return _frame_counter;
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

    b->frame_number_last_used = _frame_counter;
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
    pBlock block = write1(cache_)->find( ref );

    if (!block)
    {
        TIME_GETBLOCK TaskTimer tt(format("getBlock %s") % ReferenceInfo(ref, block_layout_, visualization_params_));

        if ( MAX_CREATED_BLOCKS_PER_FRAME > _created_count || true )
        {
            pBlock reuse;
            if (0!= "Reuse old redundant blocks")
            {
                // prefer to use block rather than discard an old block and then reallocate it
                reuse = Blocks::GarbageCollector(cache_).releaseOneBlock(_frame_counter);
            }

            BlockFactory bf(block_layout_, visualization_params_);
            block = bf.createBlock (ref, reuse);
            bool empty_cache = this->cacheCount () == 0;
            if ( !block && !empty_cache ) {
                TaskTimer tt(format("Memory allocation failed creating new block %s. Doing garbage collection") % ref);
                Blocks::GarbageCollector(cache_).releaseAllNotUsedInThisFrame (_frame_counter);
                reuse.reset ();
                block = bf.createBlock (ref, pBlock());
            }

            //Blocks::Merger(cache_).fillBlockFromOthers (block);
            merger_->fillBlockFromOthers (block);

            write1(cache_)->insert(block);

            _created_count++;
            recently_created_ |= block->getInterval ();
        }
        else
        {
            failed_allocation_ = true;
            TaskInfo(format("Delaying creation of block %s") % RegionFactory(block_layout_)(ref));
        }
    }

    return block;
}


unsigned long Collection::
        cacheByteSize() const
{
    return BlockCacheInfo::cacheByteSize (read1(cache_)->cache());
}


unsigned Collection::
        cacheCount() const
{
    return read1(cache_)->cache().size();
}


void Collection::
        printCacheSize() const
{
    BlockCacheInfo::printCacheSize(read1(cache_)->cache());
}


BlockCache::Ptr Collection::
        cache() const
{
    return cache_;
}


void Collection::
        discardOutside(Signal::Interval I)
{
    std::list<pBlock> discarded = Blocks::ClearInterval(cache_).discardOutside (I);
    BOOST_FOREACH(pBlock b, discarded) {
        removeBlock (b);
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


BlockLayout Collection::
        block_layout() const
{
    return block_layout_;
}


VisualizationParams::ConstPtr Collection::
        visualization_params() const
{
    return visualization_params_;
}


void Collection::
        length(float length)
{
    _max_sample_size.time = 2.f*std::max(1.f, length)/block_layout_.texels_per_row ();

    // If the signal has gotten shorter, make sure to discard all blocks that
    // go outside the new shorter interval
    if (_prev_length > length)
        discardOutside( Signal::Interval(0, length*block_layout_.targetSampleRate ()) );

    _prev_length = length;
}


void Collection::
        block_layout(BlockLayout v)
{
    if (block_layout_ == v)
        return;

    block_layout_ = v;

    merger_.reset( new Blocks::MergerTexture(cache_, block_layout_) );

    _max_sample_size.scale = 1.f/block_layout_.texels_per_column ();
    length(_prev_length);
    clear();
}


void Collection::
        visualization_params(VisualizationParams::ConstPtr v)
{
    EXCEPTION_ASSERT( v );

    if (visualization_params_ != v)
    {
        clear();

        visualization_params_ = v;
    }

    recently_created_ = Signal::Intervals::Intervals_ALL;
}


Intervals Collection::
        needed_samples(UnsignedIntervalType& smallest_length)
{
    Intervals r;

    if (!_is_visible)
        return r;

    BOOST_FOREACH ( const BlockCache::recent_t::value_type& a, read1(cache_)->recent() )
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


Signal::Intervals Collection::
        recently_created()
{
    Signal::Intervals r = recently_created_;
    recently_created_.clear ();
    return r;
}


void Collection::
        removeBlock (pBlock b)
{
    write1(cache_)->erase(b->reference());
    _to_remove.push_back( b );
    b->glblock.reset ();
}

} // namespace Heightmap
