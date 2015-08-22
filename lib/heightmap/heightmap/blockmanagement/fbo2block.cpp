#include "fbo2block.h"
#include "heightmap/uncaughtexception.h"
#include "heightmap/render/blocktextures.h"
#include "GlException.h"
#include "log.h"
#include "gluperspective.h"
#include "glstate.h"
#include "exceptionassert.h"

namespace Heightmap {
namespace BlockManagement {

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
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, dst->getWidth (), dst->getHeight ());
    glBindFramebuffer(GL_READ_FRAMEBUFFER, QOpenGLContext::currentContext ()->defaultFramebufferObject ());
}

void blitTexture(GlTexture::ptr src, unsigned& copyfbo)
{
    // OpenGL ES doesn't have GL_READ_FRAMEBUFFER/GL_DRAW_FRAMEBUFFER

    // Assumes dst and src have the same size and the same pixel format
    int w = src->getWidth ();
    int h = src->getHeight ();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, copyfbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, src->getOpenGlTextureId (), 0);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, QOpenGLContext::currentContext ()->defaultFramebufferObject ());
}

//void texture2texture(GlTexture::ptr src, GlTexture::ptr dst, unsigned copyfbo)
//{
//    // Assumes dst and src have the same size and the same pixel format
//    glBindFramebuffer(GL_READ_FRAMEBUFFER, copyfbo);
//    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
//                           GL_TEXTURE_2D, src->getOpenGlTextureId (), 0);
//    glBindFramebuffer(GL_READ_FRAMEBUFFER, QOpenGLContext::currentContext ()->defaultFramebufferObject ());
//    fbo2Texture(copyfbo, dst);
//}
#endif


Fbo2Block::Fbo2Block ()
{
    glGenFramebuffers(1, &drawFbo);
#ifndef GL_ES_VERSION_2_0
    glGenFramebuffers(1, &readFbo);
#endif
}


Fbo2Block::
        ~Fbo2Block()
{
    if (!QOpenGLContext::currentContext ()) {
#ifndef GL_ES_VERSION_2_0
        Log ("%s: destruction without gl context leaks fbo %d and %d") % __FILE__ % readFbo % drawFbo;
#else
        Log ("%s: destruction without gl context leaks fbo %d") % __FILE__ % drawFbo;
#endif
        return;
    }

    end();

#ifndef GL_ES_VERSION_2_0
    if (readFbo)
        glDeleteFramebuffers(1, &readFbo);
#endif
    if (drawFbo)
        glDeleteFramebuffers(1, &drawFbo);
}


Fbo2Block::ScopeBinding Fbo2Block::
        begin (Region overlapping, GlTexture::ptr srcTexture, GlTexture::ptr drawTexture, glProjection& M)
{
    EXCEPTION_ASSERT(!this->drawTexture);
    EXCEPTION_ASSERT(srcTexture);
    EXCEPTION_ASSERT(drawTexture);

    int w = drawTexture->getWidth ();
    int h = drawTexture->getHeight ();
    this->drawTexture = drawTexture;

    GlException_CHECK_ERROR ();

    // Disable unwanted capabilities when resampling a texture
    GlState::glDisable (GL_DEPTH_TEST, true); // disable depth test before binding framebuffer without depth buffer
    GlState::glDisable (GL_CULL_FACE);

    glBindTexture (GL_TEXTURE_2D, drawTexture->getOpenGlTextureId ());
#ifdef GL_ES_VERSION_2_0
    if (srcTexture!=drawTexture)
        texture2texture(srcTexture, drawTexture);

    #ifndef GL_ES_VERSION_3_0
        glBindFramebuffer(GL_FRAMEBUFFER, drawFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, drawTexture->getOpenGlTextureId (), 0);
    #else
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFbo);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, drawTexture->getOpenGlTextureId (), 0);
    #endif
#else
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFbo);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, drawTexture->getOpenGlTextureId (), 0);

    if (srcTexture!=drawTexture)
        blitTexture(srcTexture, readFbo);
#endif

    // Setup matrices
    glViewport (0, 0, w, h);
    glhOrtho (M.projection.v (), overlapping.a.time, overlapping.b.time, overlapping.a.scale, overlapping.b.scale, -10,10);
    M.modelview = matrixd::identity();
    int vp[]{0,0,w,h};
    M.viewport = vp;

    GlException_CHECK_ERROR ();

    return ScopeBinding(*this, &Fbo2Block::end);
}


void Fbo2Block::
        end()
{
    if (!drawTexture)
        return;

    // detach the texture explicitly, otherwise the texture image will not be detached if the texture is deleted
    // https://www.khronos.org/opengles/sdk/docs/man/xhtml/glFramebufferTexture2D.xml
#if defined(GL_ES_VERSION_2_0) && !defined(GL_ES_VERSION_3_0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, QOpenGLContext::currentContext ()->defaultFramebufferObject ());
#else
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, QOpenGLContext::currentContext ()->defaultFramebufferObject ());
#endif

    drawTexture.reset ();

    GlState::glEnable (GL_DEPTH_TEST);
    GlState::glEnable (GL_CULL_FACE);
}


} // namespace BlockManagement
} // namespace Heightmap
