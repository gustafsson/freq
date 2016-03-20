#include "blockinitializer.h"
#include "blockfactory.h"
#include "merge/mergertexture.h"

#include "tasktimer.h"
#include "neat_math.h"
#include "glframebuffer.h"
#include "gl.h"
#include "GlException.h"
#include "log.h"

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
      cache_(c),
      fbo_(0)
{
#ifdef DO_MERGE
    merger_.reset( new Merge::MergerTexture(cache_, bl) );
#endif
}


BlockInitializer::~BlockInitializer()
{
    if (fbo_ &&  !QOpenGLContext::currentContext ()) {
        Log ("%s: destruction without gl context leaks fbo %d") % __FILE__ % fbo_;
        return;
    }

    if (fbo_) glDeleteFramebuffers(1, &fbo_);
    fbo_ = 0;
}


Signal::Intervals BlockInitializer::
        initBlocks( const std::vector<pBlock>& blocks )
{
    TIME_GETBLOCK TaskTimer tt(format("BlockInitializer: initBlock %s") % blocks.size ());
    //TIME_GETBLOCK TaskTimer tt(format("BlockInitializer: initBlock %s") % ReferenceInfo(block->reference (), block_layout_, visualization_params_));

#ifdef DO_MERGE
    return merger_->fillBlocksFromOthers (blocks);
#else
    QOpenGLFunctions::initializeOpenGLFunctions ();
    if (!fbo_)
        glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_);
    glViewport(0, 0, block_layout_.texels_per_row (), block_layout_.texels_per_column () );
    glClearColor (0,0,0,1);
    Signal::Intervals I;
    for (const pBlock& b : blocks) {
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, b->texture ()->getOpenGlTextureId (), 0);
        glClear( GL_COLOR_BUFFER_BIT );
        I |= b->getInterval ();
    }
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, QOpenGLContext::currentContext ()->defaultFramebufferObject ());
    return I;
#endif
}

} // namespace BlockManagement
} // namespace Heightmap

#include "heightmap/render/blocktextures.h"

#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget
#include "glstate.h"

namespace Heightmap {
namespace BlockManagement {

void BlockInitializer::
        test()
{
    std::string name = "BlockInitializer";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
#ifndef LEGACY_OPENGL
    QGLFormat f = QGLFormat::defaultFormat ();
    f.setProfile( QGLFormat::CoreProfile );
    f.setVersion( 3, 2 );
    QGLFormat::setDefaultFormat (f);
#endif
    QGLWidget w;
    w.makeCurrent ();
#ifndef LEGACY_OPENGL
    GLuint VertexArrayID;
    GlException_SAFE_CALL( glGenVertexArrays(1, &VertexArrayID) );
    GlException_SAFE_CALL( glBindVertexArray(VertexArrayID) );
#endif
    GlState::assume_default_gl_states ();

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

        Render::BlockTextures::Scoped bt_raii(bl.texels_per_row (), bl.texels_per_column ());
        pBlock block = BlockFactory ().reset(bl, vp).createBlock(r);
        block_initializer.initBlock (block);
        cache->insert (block);

        EXCEPTION_ASSERT(block);
        EXCEPTION_ASSERT(cache->find(r));
        EXCEPTION_ASSERT(cache->find(r) == block);

        pBlock block3 = BlockFactory ().reset(bl, vp).createBlock(r.bottom ());
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
