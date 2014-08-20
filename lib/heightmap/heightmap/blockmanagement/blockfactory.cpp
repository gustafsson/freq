#include "blockfactory.h"
#include "tasktimer.h"

#include "GlException.h"
#include "computationkernel.h"
#include "gl.h"

//#define TIME_BLOCKFACTORY
#define TIME_BLOCKFACTORY if(0)

//#define INFO_BLOCKFACTORY
#define INFO_BLOCKFACTORY if(0)


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
      recently_created_(Signal::Intervals::Intervals_ALL),
      created_count_(0)
{
    EXCEPTION_ASSERT(visualization_params_);
}


pBlock BlockFactory::
        createBlock( const Reference& ref, GlTexture::ptr tex )
{
    TIME_BLOCKFACTORY TaskTimer tt(format("New block %s") % ReferenceInfo(ref, block_layout_, visualization_params_));

    pBlock block( new Block(
                     ref,
                     block_layout_,
                     visualization_params_,
                      tex) );

    recently_created_ |= block->getInterval ();

    //setDummyValues(block);

    return block;
}


Signal::Intervals BlockFactory::
        recently_created()
{
    Signal::Intervals r = recently_created_;
    recently_created_.clear ();
    return r;
}


void BlockFactory::
        next_frame()
{
    created_count_ = 0;
}


void BlockFactory::
        setDummyValues( pBlock block )
{
    unsigned samples = block_layout_.texels_per_row (),
            scales = block_layout_.texels_per_column ();
    std::vector<float> p(samples*scales);

    for (unsigned s = 0; s<samples; s++) {
        for (unsigned f = 0; f<scales; f++) {
            p[ f*samples + s] = 0.05f  +  0.05f * sin(s*10./samples) * cos(f*10./scales);
        }
    }

    auto ts = block->texture ()->getScopeBinding ();
    GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, samples, scales, GL_RED, GL_FLOAT, &p[0]) );
}

} // namespace BlockManagement
} // namespace Heightmap

#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget
#include "neat_math.h"

namespace Heightmap {
namespace BlockManagement {

void BlockFactory::
        test()
{
    // It should create new blocks to make them ready for receiving heightmap data and rendering.
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

        pBlock block = BlockFactory(bl, vp).createBlock(r, GlTexture::ptr());
        cache->insert (block);

        EXCEPTION_ASSERT(block);
        EXCEPTION_ASSERT(cache->find(r));
        EXCEPTION_ASSERT(cache->find(r) == block);

        pBlock block3 = BlockFactory(bl, vp).createBlock(r.bottom (), GlTexture::ptr());
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
