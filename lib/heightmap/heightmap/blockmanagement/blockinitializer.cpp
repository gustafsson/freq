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
    merger_.reset( new Merge::MergerTexture(cache_, bl) );
}


void BlockInitializer::
        initBlock( pBlock block )
{
    TIME_GETBLOCK TaskTimer tt(format("getBlock %s") % ReferenceInfo(block->reference (), block_layout_, visualization_params_));

#ifdef DO_MERGE
//    Blocks::Merger(cache_).fillBlockFromOthers (block);
    merger_->fillBlockFromOthers (block);
#else
    {
        GlFrameBuffer fbo(block->glblock->glTexture ()->getOpenGlTextureId ());
        fbo.bindFrameBuffer ();
        float v[4];
        glGetFloatv(GL_COLOR_CLEAR_VALUE, v);
        glClearColor (0,0,0,1);
        glClear (GL_COLOR_BUFFER_BIT);
        glClearColor (v[0],v[1],v[2],v[3]);
        fbo.unbindFrameBuffer ();
    }
#endif
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

        pBlock block = BlockFactory(bl, vp).createBlock(r);
        block_initializer.initBlock (block);
        cache->insert (block);

        EXCEPTION_ASSERT(block);
        EXCEPTION_ASSERT(cache->find(r));
        EXCEPTION_ASSERT(cache->find(r) == block);

        pBlock block3 = BlockFactory(bl, vp).createBlock(r.bottom ());
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
