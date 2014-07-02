#include "fbo2block.h"
#include "heightmap/uncaughtexception.h"
#include "heightmap/update/updatequeue.h"
#include "heightmap/render/blocktextures.h"
#include "GlException.h"
#include "log.h"

#include "gl.h"

const bool draw_straight_onto_block = false;

namespace Heightmap {
namespace Update {
namespace OpenGL {

void grabToTexture(GlTexture::ptr t)
{
    glBindTexture(GL_TEXTURE_2D, t->getOpenGlTextureId ());
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, t->getWidth (), t->getHeight ());
    glBindTexture(GL_TEXTURE_2D, 0);
}


void copyTexture(unsigned copyfbo, GlTexture::ptr dst, GlTexture::ptr src)
{
    // Assumes dst and src have the same size and the same pixel format
    int w = dst->getWidth ();
    int h = dst->getHeight ();
    glBindFramebuffer(GL_FRAMEBUFFER, copyfbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, src->getOpenGlTextureId (), 0);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                           GL_TEXTURE_2D, dst->getOpenGlTextureId (), 0);
    glDrawBuffer (GL_COLOR_ATTACHMENT1);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glDrawBuffer (GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


Fbo2Block::Fbo2Block ()
{
    if (!draw_straight_onto_block)
    {
        glGenFramebuffersEXT(1, &copyfbo);
        fboTexture = Render::BlockTextures(4,4,1).get1 ();
    }
}


Fbo2Block::
        ~Fbo2Block()
{
    end();
    glDeleteFramebuffers(1, &copyfbo);
}


Fbo2Block::ScopeBinding Fbo2Block::
        begin (Region br, GlTexture::ptr blockTexture)
{
    EXCEPTION_ASSERT(!this->blockTexture);
    EXCEPTION_ASSERT(blockTexture);

    int w = blockTexture->getWidth ();
    int h = blockTexture->getHeight ();
    this->blockTexture = blockTexture;

    if (draw_straight_onto_block)
    {
        fboTexture = blockTexture;
    }
    else
    {
        int oldw = fboTexture->getWidth ();
        int oldh = fboTexture->getHeight ();
        if (oldw != w || oldh != h)
        {
            int id = fboTexture->getOpenGlTextureId ();
            Render::BlockTextures::setupTexture (id, w, h);
            fboTexture.reset (new GlTexture(id));
            fbo.reset ();
        }

        copyTexture (copyfbo, fboTexture, blockTexture);
    }

    if (!fbo)
        fbo.reset (new GlFrameBuffer(fboTexture->getOpenGlTextureId ()));

    GlException_CHECK_ERROR ();

    fbo->bindFrameBuffer();
    ScopeBinding fboBinding = ScopeBinding(*this, &Fbo2Block::end);

    // Juggle texture coordinates so that border texels are centered on the border
    float dt = br.time (), ds = br.scale ();
    br.a.time -= 0.5*dt / w;
    br.b.time += 0.5*dt / w;
    br.a.scale -= 0.5*ds / h;
    br.b.scale += 0.5*ds / h;

    // Setup matrices
    glViewport (0, 0, w, h);
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



void Fbo2Block::
        end()
{
    if (!blockTexture)
        return;

    if (draw_straight_onto_block)
    {
        fbo->unbindFrameBuffer();
        fbo.reset ();
        fboTexture.reset ();
    }
    else
    {
        grabToTexture (blockTexture);
        fbo->unbindFrameBuffer();
    }

    blockTexture.reset ();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
