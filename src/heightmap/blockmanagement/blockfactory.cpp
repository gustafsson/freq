#include "blockfactory.h"
#include "tasktimer.h"
#include "heightmap/render/glblock.h"

#include "GlException.h"
#include "computationkernel.h"

//#define TIME_BLOCKFACTORY
#define TIME_BLOCKFACTORY if(0)

//#define INFO_BLOCKFACTORY
#define INFO_BLOCKFACTORY if(0)

#define MAX_CREATED_BLOCKS_PER_FRAME 16


using namespace boost;
using namespace Signal;

namespace Heightmap {
namespace BlockManagement {

BlockFactory::
        BlockFactory(BlockLayout bl, VisualizationParams::const_ptr vp)
    :
      block_layout_(bl),
      visualization_params_(vp),
      _free_memory(availableMemoryForSingleAllocation()),
      created_count_(0),
      failed_allocation_(false),
      failed_allocation_prev_(false)
{
}


pBlock BlockFactory::
        createBlock( const Reference& ref, pGlBlock reuse )
{
    pBlock b;

    if ( MAX_CREATED_BLOCKS_PER_FRAME > created_count_ || true )
    {
        if ((b = createBlockInternal (ref, reuse)))
        {
            created_count_++;
            recently_created_ |= b->getInterval ();
        }
        else
        {
            TaskInfo("failed");
        }
    }
    else
    {
        failed_allocation_ = true;

        TaskInfo(format("Delaying creation of block %s") % RegionFactory(block_layout_)(ref));
    }

    return b;
}


pBlock BlockFactory::
        createBlockInternal( const Reference& ref, pGlBlock reuse )
{
    // A precautious wrapper to getAllocatedBlock which is a precautious wrapper to attempt

    EXCEPTION_ASSERT(visualization_params_);

    TIME_BLOCKFACTORY TaskTimer tt(format("New block %s") % ReferenceInfo(ref, block_layout_, visualization_params_));

    pBlock result;
    // Try to create a new block
    try
    {
        pBlock block;
        if (reuse) {
            block.reset( new Block(
                             ref,
                             block_layout_,
                             visualization_params_) );
            block->glblock = reuse;
            // Sets to zero on first read
            block->block_data()->cpu_copy.reset( new DataStorage<float>(block->glblock->heightSize()) );
            block->discard_new_block_data();

            const Region& r = block->getRegion();
            block->glblock->reset( r.time(), r.scale() );
        }

        if (!block)
            block = getAllocatedBlock(ref);

        if (!block)
            return pBlock();

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


Signal::Intervals BlockFactory::
        recently_created()
{
    Signal::Intervals r = recently_created_;
    recently_created_.clear ();
    return r;
}


void BlockFactory::
        set_recently_created_all()
{
    recently_created_ = Signal::Intervals::Intervals_ALL;
}


bool BlockFactory::


        failed_allocation()
{
    bool r = failed_allocation_ || failed_allocation_prev_;
    failed_allocation_prev_ = failed_allocation_;
    failed_allocation_ = false;
    return r;
}


void BlockFactory::
        next_frame()
{
    created_count_ = 0;
}


pBlock BlockFactory::
        attempt( const Reference& ref )
{
    try {
        INFO_BLOCKFACTORY TaskTimer tt("Allocation attempt");

        GlException_CHECK_ERROR();
        ComputationCheckError();

        pBlock attempt( new Block( ref, block_layout_, visualization_params_ ));
        Region r = RegionFactory( block_layout_ )( ref );
        EXCEPTION_ASSERT_LESS( r.a.scale, 1 );
        EXCEPTION_ASSERT_LESS_OR_EQUAL( r.b.scale, 1 );
        attempt->glblock.reset( new GlBlock( block_layout_, r.time(), r.scale() ));

        attempt->block_data()->cpu_copy.reset( new DataStorage<float>(attempt->glblock->heightSize()) );
        attempt->discard_new_block_data();

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
    // just a precautious wrapper to BlockFactory::attempt

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

    pBlock block = attempt( ref );

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

} // namespace BlockManagement
} // namespace Heightmap

#include <QApplication>
#include <QGLWidget>
#include "neat_math.h"

namespace Heightmap {
namespace BlockManagement {

void BlockFactory::
        test()
{
    std::string name = "BlockFactory";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should create new blocks to make them ready for receiving transform data and rendering.
    {
        BlockLayout bl(4,4,4);
        VisualizationParams::const_ptr vp(new VisualizationParams);
        BlockCache::ptr cache(new BlockCache);

        Position max_sample_size;
        max_sample_size.time = 2.f / bl.texels_per_row ();
        max_sample_size.scale = 1.f / bl.texels_per_column ();

        Reference r;
        r.log2_samples_size = Reference::Scale( floor_log2( max_sample_size.time ), floor_log2( max_sample_size.scale ));
        r.block_index = Reference::Index(0,0);

        pBlock block = BlockFactory(bl, vp).createBlock(r);
        cache->insert (block);

        EXCEPTION_ASSERT(block);
        EXCEPTION_ASSERT(cache->find(r));
        EXCEPTION_ASSERT(cache->find(r) == block);

        pBlock block3 = BlockFactory(bl, vp).createBlock(r.bottom ());
        cache->insert (block3);

        EXCEPTION_ASSERT(block3);
        EXCEPTION_ASSERT(block != block3);
        EXCEPTION_ASSERT(cache->find(r.bottom ()) == block3);
    }

    // TODO test that BlockFactory behaves well on out-of-memory/reusing old blocks
    {

    }

}

} // namespace BlockManagement
} // namespace Heightmap
