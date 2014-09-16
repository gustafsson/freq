#include "fbo2block.h"
#include "heightmap/uncaughtexception.h"
#include "heightmap/update/updatequeue.h"
#include "heightmap/render/blocktextures.h"
#include "GlException.h"
#include "log.h"
#include "gluperspective.h"
#include "gl.h"

#ifdef GL_ES_VERSION_2_0
const bool copy_to_new_fbo_for_each_draw = true;
#else
const bool copy_to_new_fbo_for_each_draw = false;
#endif

namespace Heightmap {
namespace Update {
namespace OpenGL {


#ifdef GL_ES_VERSION_2_0
void texture2texture(GlTexture::ptr src, GlTexture::ptr dst)
{
    GlException_SAFE_CALL( glCopyTextureLevelsAPPLE(dst->getOpenGlTextureId (), src->getOpenGlTextureId (), 0, 1) );
}
#else
void fbo2Texture(unsigned fbo, GlTexture::ptr dst)
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glBindTexture(GL_TEXTURE_2D, dst->getOpenGlTextureId ());
    GlException_SAFE_CALL( glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, dst->getWidth (), dst->getHeight ()) );
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void blitTexture(GlTexture::ptr src, unsigned& copyfbo)
{
    // opengles doesn't have GL_READ_FRAMEBUFFER/GL_WRITE_FRAMEBUFFER

    // Assumes dst and src have the same size and the same pixel format
    int w = src->getWidth ();
    int h = src->getHeight ();
    if (!copyfbo)
        glGenFramebuffers(1, &copyfbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, copyfbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, src->getOpenGlTextureId (), 0);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void texture2texture(GlTexture::ptr src, GlTexture::ptr dst, unsigned copyfbo)
{
    // Assumes dst and src have the same size and the same pixel format
    glBindFramebuffer(GL_READ_FRAMEBUFFER, copyfbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, src->getOpenGlTextureId (), 0);
    fbo2Texture(copyfbo, dst);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}
#endif


Fbo2Block::Fbo2Block ()
{
    if (!copy_to_new_fbo_for_each_draw)
    {
        fboTexture = Render::BlockTextures::get1 ();
        fbo.reset (new GlFrameBuffer(*fboTexture));
    }

    glGenFramebuffers(1, &copyfbo);
}


Fbo2Block::
        ~Fbo2Block()
{
    end();

    if (copyfbo)
        glDeleteFramebuffers(1, &copyfbo);
}


Fbo2Block::ScopeBinding Fbo2Block::
        begin (Region br, GlTexture::ptr oldTexture, GlTexture::ptr targetTexture, glProjection& M)
{
    EXCEPTION_ASSERT(!this->targetTexture);
    EXCEPTION_ASSERT(targetTexture);

    int w = targetTexture->getWidth ();
    int h = targetTexture->getHeight ();
    this->targetTexture = targetTexture;

    if (copy_to_new_fbo_for_each_draw)
    {
        fboTexture = targetTexture;
        fbo.reset (new GlFrameBuffer(*fboTexture));
    }

    GlException_CHECK_ERROR ();

#ifdef GL_ES_VERSION_2_0
    texture2texture(oldTexture, fboTexture);
    fbo->bindFrameBuffer();
#else
    fbo->bindFrameBuffer();
    blitTexture(oldTexture, copyfbo);
#endif

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
    if (!targetTexture)
        return;

    fbo->unbindFrameBuffer();

    if (copy_to_new_fbo_for_each_draw)
    {
        // The fbo was created with targetTexture as color attachment
        fbo.reset ();
        fboTexture.reset ();
    }
    else
    {
#ifdef GL_ES_VERSION_2_0
        texture2texture (fboTexture, targetTexture);
#else
        fbo2Texture(fbo->getOpenGlFboId(), targetTexture);
#endif
    }

    targetTexture.reset ();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
