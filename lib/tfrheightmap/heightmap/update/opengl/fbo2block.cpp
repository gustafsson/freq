#include "fbo2block.h"
#include "heightmap/render/glblock.h"
#include "heightmap/uncaughtexception.h"
#include "GlException.h"

#include "gl.h"

namespace Heightmap {
namespace Update {
namespace OpenGL {

void grabToTexture(GlTexture::ptr t)
{
    glBindTexture(GL_TEXTURE_2D, t->getOpenGlTextureId ());
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, t->getWidth (), t->getHeight ());
    glBindTexture(GL_TEXTURE_2D, 0);
}


Fbo2Block::Fbo2Block (pBlock block)
    :
      block(block),
      glblock(block->glblock)
{
    // Create new texture
//    glblock.reset(new GlBlock(block->block_layout(), block->getRegion().time (), block->getRegion().scale ()));
    fbo.reset (new GlFrameBuffer(glblock->glTexture ()->getOpenGlTextureId ()));

    // Copy from texture to own fbo
//    {
//        GlFrameBuffer fbo(block->glblock->glTexture ()->getOpenGlTextureId ());
//        fbo.bindFrameBuffer ();
//        grabToTexture(glblock->glTexture());
//        fbo.unbindFrameBuffer ();
//    }
}


Fbo2Block::~Fbo2Block()
{
    if (!fbo)
        return;

    try {
//        fbo->bindFrameBuffer ();
//        grabToTexture(block->glblock->glTexture());
//        grabToTexture(block->glblock->glVertTexture());
//        fbo->unbindFrameBuffer ();

        // Discard previous glblock ... wrong thread ... could also grabToTexture into oldglblock
        fbo->bindFrameBuffer ();
        grabToTexture(glblock->glVertTexture());
        fbo->unbindFrameBuffer ();
//        glFinish ();
//        block->glblock = glblock;

        block->discard_new_block_data ();
    } catch (...) {
        Heightmap::UncaughtException::handle_exception(boost::current_exception());
    }
}

GlFrameBuffer::ScopeBinding Fbo2Block::
        begin ()
{
    GlException_CHECK_ERROR ();

    GlFrameBuffer::ScopeBinding fboBinding = fbo->getScopeBinding ();
//        GlTexture::ScopeBinding texObjBinding = chunk_texture_->getScopeBinding();
//        GlException_SAFE_CALL( glDrawArrays(GL_TRIANGLE_STRIP, 0, nScales*2) );

    Region br = block->getRegion ();
    BlockLayout block_layout = block->block_layout ();

    // Juggle texture coordinates so that border texels are centered on the border
    float dt = br.time (), ds = br.scale ();
    br.a.time -= 0.5*dt / block_layout.texels_per_row ();
    br.b.time += 0.5*dt / block_layout.texels_per_row ();
    br.a.scale -= 0.5*ds / block_layout.texels_per_column ();
    br.b.scale += 0.5*ds / block_layout.texels_per_column ();

    // Setup matrices
    glViewport (0, 0, block_layout.texels_per_row (), block_layout.texels_per_column ());
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    glOrtho (br.a.time, br.b.time, br.a.scale, br.b.scale, -10,10);
    glMatrixMode (GL_MODELVIEW);
    glLoadIdentity ();

    GlException_CHECK_ERROR ();

    // Disable unwanted capabilities when resampling a texture
    glDisable (GL_DEPTH_TEST);
    glDisable (GL_BLEND);
    glDisable (GL_CULL_FACE);

    return fboBinding;
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
