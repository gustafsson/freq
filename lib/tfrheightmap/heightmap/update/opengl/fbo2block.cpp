#include "fbo2block.h"
#include "heightmap/uncaughtexception.h"
#include "heightmap/update/updatequeue.h"
#include "heightmap/render/blocktextures.h"
#include "GlException.h"
#include "log.h"
#include "gluperspective.h"
#include "gl.h"

#ifdef GL_ES_VERSION_2_0
const bool draw_straight_onto_block = true;
#else
const bool draw_straight_onto_block = false;
#endif

namespace Heightmap {
namespace Update {
namespace OpenGL {

#ifdef GL_ES_VERSION_2_0
void grabToTexture(GlTexture::ptr dst, GlTexture::ptr src)
{
    GlException_SAFE_CALL( glCopyTextureLevelsAPPLE(dst->getOpenGlTextureId (), src->getOpenGlTextureId (), 0, 1) );
}
#else
void grabToTexture(GlTexture::ptr dst)
{
    glBindTexture(GL_TEXTURE_2D, dst->getOpenGlTextureId ());
    GlException_SAFE_CALL( glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, dst->getWidth (), dst->getHeight ()) );
    glBindTexture(GL_TEXTURE_2D, 0);
}
#endif


void copyTexture(unsigned& copyfbo, GlTexture::ptr dst, GlTexture::ptr src)
{
#ifdef GL_ES_VERSION_2_0
    GlException_SAFE_CALL( glCopyTextureLevelsAPPLE(dst->getOpenGlTextureId (), src->getOpenGlTextureId (), 0, 1) );
#else
    // Assumes dst and src have the same size and the same pixel format
    int w = dst->getWidth ();
    int h = dst->getHeight ();
    if (!copyfbo)
        glGenFramebuffers(1, &copyfbo);
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
#endif
}


Fbo2Block::Fbo2Block ()
{
}


Fbo2Block::
        ~Fbo2Block()
{
    end();

    if (copyfbo)
        glDeleteFramebuffers(1, &copyfbo);
}


Fbo2Block::ScopeBinding Fbo2Block::
        begin (Region br, GlTexture::ptr blockTexture, glProjection& M)
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
        int oldw = fboTexture ? fboTexture->getWidth () : -1;
        int oldh = fboTexture ? fboTexture->getHeight () : -1;
        if (oldw != w || oldh != h)
        {
            fbo.reset ();
            fboTexture.reset ();
            fboTexture = Render::BlockTextures(w,h,1).get1 ();
        }

        copyTexture (copyfbo, fboTexture, blockTexture);
    }

    if (!fbo)
        fbo.reset (new GlFrameBuffer(*fboTexture));

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
    glhOrtho (M.projection.v (), br.a.time, br.b.time, br.a.scale, br.b.scale, -10,10);
    M.modelview = matrixd::identity();
    int vp[]{0,0,w,h};
    M.viewport = vp;

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
#ifdef GL_ES_VERSION_2_0
        fbo->unbindFrameBuffer();
        grabToTexture (blockTexture, fboTexture);
#else
        grabToTexture (blockTexture);
        fbo->unbindFrameBuffer();
#endif
    }

    blockTexture.reset ();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
