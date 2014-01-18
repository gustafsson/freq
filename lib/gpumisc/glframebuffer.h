/**
See class comment GlFrameBuffer.
*/

#pragma once

#include "GlTexture.h"

/**
GlFrameBuffer is a wrapper for an OpenGL frame buffer object (FBO) (refered to
as just "frame buffer") to manage the frame buffer in an object oriented manner
(mainly construction and destruction and keeping track of FBO id).

@author johan.b.gustafsson@gmail.com
*/
class GlFrameBuffer: public boost::noncopyable
{
public:
    typedef ReleaseAfterContext<GlFrameBuffer> TextureBinding;

    /**
    Creates a new OpenGL frame buffer object and allocates memory. Width and
    height are taken from the current viewport if width is 0.

    @throws GlException If OpenGL encountered an error.
    */
    GlFrameBuffer(unsigned width=0, unsigned height=0);
    GlFrameBuffer(GlTexture* texture);

    /**
    Releases the frame buffer object.
    */
    ~GlFrameBuffer();

    typedef ReleaseAfterContext<GlFrameBuffer> ScopeBinding;

    /**
    Binds this frame buffer with glBindFramebufferEXT. Then removes the binding
    when the object goes out of scope.
    */
    ScopeBinding getScopeBinding() const;
    //ScopeBinding doOffscreenRenderingInCallersScope() const { return getScopeBinding(); }

    /**
    Binds this frame buffer with glBindFramebufferEXT.
    */
    void bindFrameBuffer() const;

    /**
    Removes the binding with glBindFramebufferEXT.
    */
    void unbindFrameBuffer() const;

    /**
    Returns the OpenGL frame buffer object id for this frame buffer. To be used
    with OpenGL functions such as glBindFramebufferEXT().
    */
    unsigned int getOpenGlFboId() const { return fboId_; }

    /**
    The frame buffer can be accessed through a texture.
    */
    GlTexture& getGlTexture() { return *texture_; }

    /**
    Recreates the OpenGL frame buffer object and allocates memory for a
    different size. Width and height are taken from the current viewport if
    width is 0.

    If width and height are equal to current width and height this function
    does nothing.

    @throws GlException If OpenGL encountered an error.
      */
    void recreate(unsigned width=0, unsigned height=0);

private:
    /**
    OpenGL frame buffer object id for the frame buffer.
    */
    unsigned int fboId_;

    /**
    OpenGL render buffer to bind to the depth buffer in the frame buffer.
    */
    unsigned int rboId_;

    /**
      Texture to access the frame buffer.
      */
    GlTexture* own_texture_;
    GlTexture* texture_;

    void init();

public:
    static void test();
};
