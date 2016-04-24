/**
See class comment GlFrameBuffer.
*/

#pragma once

#include "GlTexture.h"
#include <QOpenGLFunctions>

/**
GlFrameBuffer is a wrapper for an OpenGL frame buffer object (FBO) (refered to
as just "frame buffer") to manage the frame buffer in an object oriented manner
(mainly construction and destruction and keeping track of FBO id).

@author johan.b.gustafsson@gmail.com
*/
class GlFrameBuffer: public boost::noncopyable, private QOpenGLFunctions
{
public:
    typedef ReleaseAfterContext<GlFrameBuffer> TextureBinding;

    /**
    Creates a new OpenGL frame buffer object and allocates memory.

    @throws GlException If OpenGL encountered an error.
    */
    GlFrameBuffer(int width, int height);
    GlFrameBuffer(unsigned textureid, int width, int height, int level=0);
    GlFrameBuffer(const GlTexture& texture, int level=0);

    /**
    Releases the frame buffer object.
    */
    ~GlFrameBuffer();

    typedef ReleaseAfterContext<GlFrameBuffer> ScopeBinding;

    /**
    Binds this frame buffer with glBindFramebufferEXT. Then removes the binding
    when the object goes out of scope.
    */
    ScopeBinding getScopeBinding();

    /**
    Binds this frame buffer with glBindFramebufferEXT.
    */
    void bindFrameBuffer();

    /**
    Removes the binding with glBindFramebufferEXT.
    */
    void unbindFrameBuffer();

    /**
    Returns the OpenGL frame buffer object id for this frame buffer. To be used
    with OpenGL functions such as glBindFramebufferEXT().
    */
    unsigned int getOpenGlFboId() const { return fboId_; }

    /**
    The frame buffer can be accessed through a texture.
    */
    unsigned getGlTexture() { return textureid_; }

    /**
    Recreates the OpenGL frame buffer object and allocates memory for a
    different size.

    If width and height are equal to current width and height this function
    does nothing.

    @throws GlException If OpenGL encountered an error.
      */
    void recreate(int width, int height);

    /**
     The width of the texture being rendered onto.
     */
    int getWidth() { return texture_width_; }

    /**
     The height of the texture being rendered onto.
     */
    int getHeight() { return texture_height_; }

private:

    /**
    OpenGL frame buffer object id for the frame buffer.
    */
    unsigned int fboId_ = 0;

    /**
    OpenGL render buffer to bind to the depth buffer in the frame buffer.
    */
    unsigned int depth_stencil_buffer_ = 0;

    /**
    Used by (un)bindFrameBuffer to restore the state after binding.
     */
    int prev_fbo_ = 0;

    /**
      Texture to access the frame buffer.
      */
    GlTexture* own_texture_ = 0;
    unsigned textureid_ = 0;
    bool enable_depth_component_ = false;

    int texture_width_ = 0, texture_height_ = 0, level_ = 0;

public:
    static void test();
};
