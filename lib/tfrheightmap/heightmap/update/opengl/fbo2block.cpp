#include "fbo2block.h"
#include "heightmap/render/glblock.h"
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
        unsigned textureid;
        glGenTextures (1, &textureid);
        Render::BlockTextures::setupTexture (textureid,4,4);
        texture.reset (new GlTexture(textureid));
    }
}


Fbo2Block::
        ~Fbo2Block()
{
    end();
    glDeleteFramebuffers(1, &copyfbo);
}


Fbo2Block::ScopeBinding Fbo2Block::
        begin (Region br, Block::pGlBlock glblock)
{
    EXCEPTION_ASSERT(!this->glblock);
    EXCEPTION_ASSERT(glblock);

    int w = glblock->glTexture ()->getWidth ();
    int h = glblock->glTexture ()->getHeight ();
    this->glblock = glblock;

    if (draw_straight_onto_block)
    {
        texture = glblock->glTexture ();
    }
    else
    {
        int oldw = texture->getWidth ();
        int oldh = texture->getHeight ();
        if (oldw != w || oldh != h)
        {
            int id = texture->getOpenGlTextureId ();
            Render::BlockTextures::setupTexture (id, w, h);
            texture.reset (new GlTexture(id));
            fbo.reset ();
        }

        copyTexture (copyfbo, texture, glblock->glTexture ());
    }

    if (!fbo)
        fbo.reset (new GlFrameBuffer(texture->getOpenGlTextureId ()));

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
    if (!glblock)
        return;

    if (draw_straight_onto_block)
    {
        fbo->unbindFrameBuffer();
        fbo.reset ();
    }
    else
    {
        grabToTexture (glblock->glTexture ());
        fbo->unbindFrameBuffer();
    }

    glblock.reset ();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
