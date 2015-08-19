#include "GlTexture.h"

#include "GlException.h"
#include "gl.h"
#include "exceptionassert.h"

GlTexture::GlTexture()
:	width( 0 ),
    height( 0 ),
    textureId( 0 ),
    ownTextureId( 0 )
{
    reset(0, 0);
}

GlTexture::GlTexture(unsigned short width, unsigned short height)
:	width( width ),
    height( height ),
    textureId( 0 ),
    ownTextureId( 0 )
{
    reset(width, height);
}

GlTexture::GlTexture(unsigned short width, unsigned short height,
                     unsigned int pixelFormat, unsigned int internalFormat,
                     unsigned type, void* data )
:	width( width ),
    height( height ),
    textureId( 0 ),
    ownTextureId( 0 )
{
    reset(width, height, pixelFormat, internalFormat, type, data);
}

GlTexture::GlTexture(unsigned int textureId, int width, int height, bool adopt)
    :	width( width ),
        height( height ),
        textureId( textureId ),
        ownTextureId( adopt ? textureId : 0 )
{
    EXCEPTION_ASSERT_LESS(0u, textureId);
    EXCEPTION_ASSERT_LESS(0, width);
    EXCEPTION_ASSERT_LESS(0, height);
}

void GlTexture::
        reset(unsigned short width, unsigned short height,
              unsigned int pixelFormat, unsigned int internalFormat,
              unsigned type, void* data)
{
    if (0==textureId)
    {
        GlException_SAFE_CALL( glGenTextures(1, &ownTextureId) );
        textureId = ownTextureId;
    }
    this->width = width;
    this->height = height;

    if (0==width)
        return;

    glBindTexture( GL_TEXTURE_2D, textureId);

	GlException_SAFE_CALL( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE) );
	GlException_SAFE_CALL( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE) );
    //GlException_SAFE_CALL( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR) );
    GlException_SAFE_CALL( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) );
    GlException_SAFE_CALL( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) );
    int gl_max_texture_size = 0;
    glGetIntegerv (GL_MAX_TEXTURE_SIZE, &gl_max_texture_size);
    EXCEPTION_ASSERT_LESS(width, gl_max_texture_size);
    EXCEPTION_ASSERT_LESS(height, gl_max_texture_size);
    GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, pixelFormat, type, data) );
}

GlTexture::~GlTexture() {
    if (ownTextureId) {
        if (!QOpenGLContext::currentContext ()) {
            Log ("%s: destruction without gl context leaks tex %d") % __FILE__ % ownTextureId;
            return;
        }

        glDeleteTextures(1, &ownTextureId);
    }
}

void GlTexture::bindTexture()
{
    GlException_SAFE_CALL( glBindTexture( GL_TEXTURE_2D, textureId) );
}
