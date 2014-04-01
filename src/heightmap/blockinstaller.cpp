#include "blockinstaller.h"
#include "blockfactory.h"
#include "blocks/garbagecollector.h"
#include "blocks/merger.h"
#include "blocks/mergertexture.h"
#include "blocks/chunkmerger.h"

#include "tasktimer.h"
#include "neat_math.h"

//#define TIME_GETBLOCK
#define TIME_GETBLOCK if(0)

#define MAX_CREATED_BLOCKS_PER_FRAME 16

using namespace boost;

namespace Heightmap {

/*class UnlockAndLockUpgrade {
public:
    UnlockAndLockUpgrade(shared_mutex* m) : m(m) {
        m->unlock_and_lock_upgrade ();
    }

    ~UnlockAndLockUpgrade() {
        if (m)
            restore();
    }

    void restore() {
        m->unlock_upgrade_and_lock ();
        m = 0;
    }

private:
    shared_mutex* m;
};*/


BlockInstaller::
        BlockInstaller(BlockLayout bl, VisualizationParams::ConstPtr vp, BlockCache::Ptr c)
    :
      block_layout_(bl),
      visualization_params_(vp),
      cache_(c),
      created_count_(0),
      failed_allocation_(false),
      failed_allocation_prev_(false)
{
    block_layout (bl);
}



void BlockInstaller::
        block_layout(BlockLayout v)
{
    block_layout_ = v;
    merger_.reset( new Blocks::MergerTexture(cache_, v) );
}


Signal::Intervals BlockInstaller::
        recently_created()
{
    Signal::Intervals r = recently_created_;
    recently_created_.clear ();
    return r;
}


void BlockInstaller::
        set_recently_created_all()
{
    recently_created_ = Signal::Intervals::Intervals_ALL;
}


bool BlockInstaller::
        failed_allocation()
{
    bool r = failed_allocation_ || failed_allocation_prev_;
    failed_allocation_prev_ = failed_allocation_;
    failed_allocation_ = false;
    return r;
}


void BlockInstaller::
        next_frame()
{
    created_count_ = 0;
}


pBlock BlockInstaller::
        getBlock( const Reference& ref, int frame_counter )
{
    // shared_state doesn't use the 'upgrade ownership' concept, but it could use a mutex that supports it (boost::shared_mutex does for instance).
//    UnlockAndLockUpgrade unlock_and_lock_upgrade(this->readWriteLock ());

    // Look among cached blocks for this reference
    pBlock block = cache_.write ()->find( ref );

    if (!block)
    {
        TIME_GETBLOCK TaskTimer tt(format("getBlock %s") % ReferenceInfo(ref, block_layout_, visualization_params_));

        if ( MAX_CREATED_BLOCKS_PER_FRAME > created_count_ || true )
        {
            pBlock reuse;
            if (0 != "Reuse old redundant blocks")
            {
                // prefer to use block rather than discard an old block and then reallocate it
                reuse = Blocks::GarbageCollector(cache_).releaseOneBlock (frame_counter);
            }

            BlockFactory bf(block_layout_, visualization_params_);
            block = bf.createBlock (ref, reuse);
            bool empty_cache = cache_.read ()->cache().empty();
            if ( !block && !empty_cache ) {
                TaskTimer tt(format("Memory allocation failed creating new block %s. Doing garbage collection") % ref);
                Blocks::GarbageCollector(cache_).releaseAllNotUsedInThisFrame (frame_counter);
                reuse.reset ();
                block = bf.createBlock (ref, pBlock());
            }

            //Blocks::Merger(cache_).fillBlockFromOthers (block);
            merger_->fillBlockFromOthers (block);

            cache_.write ()->insert(block);

//            unlock_and_lock_upgrade.restore ();

            created_count_++;
            recently_created_ |= block->getInterval ();
        }
        else
        {
//            unlock_and_lock_upgrade.restore ();

            failed_allocation_ = true;

            TaskInfo(format("Delaying creation of block %s") % RegionFactory(block_layout_)(ref));
        }
    }

    return block;
}

} // namespace Heightmap


#include <QGLWidget>
#include <QApplication>

namespace Heightmap {

void BlockInstaller::
        test()
{
    std::string name = "BlockInstaller";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should create new blocks and install them in a cache.
    {
        BlockLayout bl(4,4,4);
        VisualizationParams::ConstPtr vp(new VisualizationParams);
        BlockCache::Ptr cache(new BlockCache);

        BlockInstaller block_installer(bl, vp, cache);

        Position max_sample_size;
        max_sample_size.time = 2.f / bl.texels_per_row ();
        max_sample_size.scale = 1.f / bl.texels_per_column ();

        Reference r;
        r.log2_samples_size = Reference::Scale( floor_log2( max_sample_size.time ), floor_log2( max_sample_size.scale ));
        r.block_index = Reference::Index(0,0);

        pBlock block = block_installer.getBlock (r, 0);

        EXCEPTION_ASSERT(block);
        EXCEPTION_ASSERT(cache.read ()->probe(r));
        EXCEPTION_ASSERT(cache.read ()->probe(r) == block);

        pBlock block2 = block_installer.getBlock (r, 0);

        EXCEPTION_ASSERT(block2);
        EXCEPTION_ASSERT(block == block2);

        pBlock block3 = block_installer.getBlock (r.bottom (), 0);

        EXCEPTION_ASSERT(block3);
        EXCEPTION_ASSERT(block != block3);
        EXCEPTION_ASSERT(cache.read ()->probe(r.bottom ()) == block3);
    }

    // TODO test that BlockInstaller behaves well on out-of-memory/reusing old blocks
    {

    }

    // TODO test that BlockInstaller fills new block with content from old blocks
    {

    }
}

} // namespace Heightmap
