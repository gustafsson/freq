#include "blockinitializer.h"
#include "blockfactory.h"
#include "merge/mergertexture.h"

#include "tasktimer.h"
#include "neat_math.h"
#include "glframebuffer.h"
#include "gl.h"

//#define TIME_GETBLOCK
#define TIME_GETBLOCK if(0)

#define DO_MERGE

using namespace boost;

namespace Heightmap {
namespace BlockManagement {

BlockInitializer::
        BlockInitializer(BlockLayout bl, VisualizationParams::const_ptr vp, BlockCache::const_ptr c)
    :
      block_layout_(bl),
      visualization_params_(vp),
      cache_(c)
{
#ifdef DO_MERGE
    merger_.reset( new Merge::MergerTexture(cache_, bl) );
#else
    bool disable_merge = true;
    merger_.reset( new Merge::MergerTexture(cache_, bl, disable_merge) );
#endif
}


void BlockInitializer::
        initBlocks( const std::vector<pBlock>& blocks )
{
    TIME_GETBLOCK TaskTimer tt(format("BlockInitializer: initBlock %s") % blocks.size ());
    //TIME_GETBLOCK TaskTimer tt(format("BlockInitializer: initBlock %s") % ReferenceInfo(block->reference (), block_layout_, visualization_params_));

    merger_->fillBlocksFromOthers (blocks);
}

} // namespace BlockManagement
} // namespace Heightmap


#include <QGLWidget>
#include <QApplication>

namespace Heightmap {
namespace BlockManagement {

void BlockInitializer::
        test()
{
    std::string name = "BlockInitializer";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should initialize new blocks
    {
        BlockLayout bl(4,4,4);
        VisualizationParams::const_ptr vp(new VisualizationParams);
        BlockCache::ptr cache(new BlockCache);

        BlockInitializer block_initializer(bl, vp, cache);

        Position max_sample_size;
        max_sample_size.time = 2.f / bl.texels_per_row ();
        max_sample_size.scale = 1.f / bl.texels_per_column ();

        Reference r;
        r.log2_samples_size = Reference::Scale( floor_log2( max_sample_size.time ), floor_log2( max_sample_size.scale ));
        r.block_index = Reference::Index(0,0);

        GlTexture::ptr tex(new GlTexture(128,128));
        pBlock block = BlockFactory(bl, vp).createBlock(r, tex);
        block_initializer.initBlock (block);
        cache->insert (block);

        EXCEPTION_ASSERT(block);
        EXCEPTION_ASSERT(cache->find(r));
        EXCEPTION_ASSERT(cache->find(r) == block);

        pBlock block3 = BlockFactory(bl, vp).createBlock(r.bottom (), tex);
        block_initializer.initBlock (block3);
        cache->insert (block3);

        EXCEPTION_ASSERT(block3);
        EXCEPTION_ASSERT(block != block3);
        EXCEPTION_ASSERT(cache->find(r.bottom ()) == block3);
    }

    // TODO test that BlockInitializer fills new block with content from old blocks
    {

    }
}

} // namespace BlockManagement
} // namespace Heightmap
