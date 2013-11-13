#include "GlTexture.h"

#include "GlException.h"
#include "gl.h"

GlTexture::GlTexture(unsigned short width, unsigned short height,
                     unsigned int pixelFormat, unsigned int internalFormat,
                     unsigned type, void* data )
:	width( width ),
    height( height ),
    pixelFormat( pixelFormat ),
    textureId( 0 )
{
    if (width!=0)
        reset(width, height, pixelFormat, internalFormat, type, data);
}

void GlTexture::
        reset(unsigned short width, unsigned short height,
              unsigned int pixelFormat, unsigned int internalFormat,
              unsigned type, void* data)
{
    if (0==textureId)
        GlException_SAFE_CALL( glGenTextures(1, &textureId) );

	bindTexture2D();

	GlException_SAFE_CALL( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE) );
	GlException_SAFE_CALL( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE) );
    //GlException_SAFE_CALL( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR) );
    GlException_SAFE_CALL( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) );
    GlException_SAFE_CALL( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) );
    GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, pixelFormat, type, data) );

    this->width = width;
    this->height = height;
    this->pixelFormat = pixelFormat;

	unbindTexture2D();
}

GlTexture::~GlTexture() {
    // GlException_CHECK_ERROR(); // Windows generates some error prior to this call, why?
	//GlException_SAFE_CALL( glDeleteTextures(1, &textureId) );
	glDeleteTextures(1, &textureId); // Any GL call "seems to be" an invalid operation on Windows after/during destruction. Might be the result of something else that is broken...
    glGetError();

	textureId = 0;
}

GlTexture::ScopeBinding GlTexture::getScopeBinding() const
{
    bindTexture2D();
    return ScopeBinding(*this, &GlTexture::unbindTexture2Dwrap);
}

void GlTexture::bindTexture2D() const {
	GlException_SAFE_CALL( glEnable(GL_TEXTURE_2D) );
	GlException_SAFE_CALL( glBindTexture( GL_TEXTURE_2D, textureId) );
}

void GlTexture::unbindTexture2D() {
	GlException_SAFE_CALL( glBindTexture( GL_TEXTURE_2D, 0) );
	GlException_SAFE_CALL( glDisable(GL_TEXTURE_2D) );
}

void GlTexture::unbindTexture2Dwrap() const {
    unbindTexture2D();
}
