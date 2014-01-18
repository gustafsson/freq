#include "glframebuffer.h"

#include "GlException.h"
#include "gl.h"
#include "exceptionassert.h"
#include "TaskTimer.h"
#include "backtrace.h"

//#define DEBUG_INFO
#define DEBUG_INFO if(0)

class GlFrameBufferException: virtual public boost::exception, virtual public std::exception {};

GlFrameBuffer::
        GlFrameBuffer(unsigned width, unsigned height)
            :
            fboId_(0),
            rboId_(0),
            own_texture_(new GlTexture),
            texture_(own_texture_)
{
    init();

    try
    {
        recreate(width, height);
    }
    catch(...)
    {
        TaskInfo("GlFrameBuffer() caught exception");
        if (rboId_) glDeleteRenderbuffersEXT(1, &rboId_);
        if (fboId_) glDeleteFramebuffersEXT(1, &fboId_);

        throw;
    }
}

GlFrameBuffer::
        GlFrameBuffer(GlTexture* texture)
    :
    fboId_(0),
    rboId_(0),
    own_texture_(0),
    texture_(texture)
{
    init();

    try
    {
        recreate(texture->getWidth (), texture->getHeight ());
    }
    catch(...)
    {
        TaskInfo("GlFrameBuffer() caught exception");
        if (rboId_) glDeleteRenderbuffersEXT(1, &rboId_);
        if (fboId_) glDeleteFramebuffersEXT(1, &fboId_);

        throw;
    }
}

GlFrameBuffer::
        ~GlFrameBuffer()
{
    DEBUG_INFO TaskTimer tt("~GlFrameBuffer()");

    GLenum e = glGetError();
    if (e == GL_INVALID_OPERATION)
    {
        TaskInfo("glGetError = GL_INVALID_OPERATION");
    }

    DEBUG_INFO TaskInfo("glGetError = %u", (unsigned)e);

    DEBUG_INFO TaskInfo("glDeleteRenderbuffersEXT");
    glDeleteRenderbuffersEXT(1, &rboId_);

    DEBUG_INFO TaskInfo("glDeleteFramebuffersEXT");
    glDeleteFramebuffersEXT(1, &fboId_);

    GLenum e2 = glGetError();
    DEBUG_INFO TaskInfo("glGetError = %u", (unsigned)e2);

    if (own_texture_)
        delete own_texture_;
}

GlFrameBuffer::ScopeBinding GlFrameBuffer::
        getScopeBinding() const
{
    bindFrameBuffer();
    return ScopeBinding(*this, &GlFrameBuffer::unbindFrameBuffer);
}

void GlFrameBuffer::
        bindFrameBuffer() const
{
    GlException_CHECK_ERROR();

    GlException_SAFE_CALL( glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboId_));

    GlException_CHECK_ERROR();
}

void GlFrameBuffer::
        unbindFrameBuffer() const
{
    GlException_CHECK_ERROR();

    GlException_SAFE_CALL( glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0) );

    GlException_CHECK_ERROR();
}


void GlFrameBuffer::
        recreate(unsigned width, unsigned height)
{
    const char* action = "Resizing";
    if (0==width)
    {
        action = "Creating";

        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        width = viewport[2];
        height = viewport[3];

        GLint intmax;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &intmax);

        if (width>(unsigned)intmax || height>(unsigned)intmax || width == 0 || height == 0)
            throw std::logic_error("Can't call GlFrameBuffer when no valid viewport is active");
    }

    if (width == texture_->getWidth() && height == texture_->getHeight() && rboId_ && fboId_)
        return;

    DEBUG_INFO TaskTimer tt("%s fbo(%u, %u)", action, width, height);

    // if (rboId_) { glDeleteRenderbuffersEXT(1, &rboId_); rboId_ = 0; }
    // if (fboId_) { glDeleteFramebuffersEXT(1, &fboId_); fboId_ = 0; }

    if (width != texture_->getWidth() || height != texture_->getHeight())
        texture_->reset(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);

    GlException_CHECK_ERROR();

    {
        if (!rboId_)
            glGenRenderbuffersEXT(1, &rboId_);

        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, rboId_);
        glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, width, height);
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
    }

    {
        if (!fboId_)
            glGenFramebuffersEXT(1, &fboId_);

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboId_);

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
                                  GL_COLOR_ATTACHMENT0_EXT,
                                  GL_TEXTURE_2D,
                                  texture_->getOpenGlTextureId(),
                                  0);

        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT,
                                     GL_DEPTH_ATTACHMENT_EXT,
                                     GL_RENDERBUFFER_EXT,
                                     rboId_);

        int status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

        if (GL_FRAMEBUFFER_UNSUPPORTED_EXT == status)
          {
            BOOST_THROW_EXCEPTION(GlFrameBufferException() << errinfo_format(boost::format(
                    "Got GL_FRAMEBUFFER_UNSUPPORTED_EXT. See GlFrameBuffer::test for supported formats")) << Backtrace::make ());
          }

        EXCEPTION_ASSERT_EQUALS( GL_FRAMEBUFFER_COMPLETE_EXT, status );

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    }

    //glFlush();

    DEBUG_INFO TaskInfo("fbo = %u", fboId_ );
    DEBUG_INFO TaskInfo("rbo = %u", fboId_ );
    DEBUG_INFO TaskInfo("texture = %u", texture_->getOpenGlTextureId() );

    GlException_CHECK_ERROR();
}


void GlFrameBuffer::
        init()
{
#ifndef __APPLE__ // glewInit is not needed on Mac
    if (0==glGenRenderbuffersEXT)
    {
        DEBUG_INFO TaskInfo("Initializing glew");

        if (0 != glewInit() )
            BOOST_THROW_EXCEPTION(GlFrameBufferException() << errinfo_format(boost::format(
                    "Couldn't initialize \"glew\"")) << Backtrace::make ());

        if (!glewIsSupported( "GL_EXT_framebuffer_object" )) {
            BOOST_THROW_EXCEPTION(GlFrameBufferException() << errinfo_format(boost::format(
                    "Failed to get minimal extensions\n"
                    "Sonic AWE requires:\n"
                    "  GL_EXT_framebuffer_object\n")) << Backtrace::make ());
        }
    }
#endif
}


//////// test //////

#include "expectexception.h"
#include <QGLWidget>
#include <QApplication>

void GlFrameBuffer::
        test()
{
    int argc = 0;
    char* argv = 0;
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should wrapper an OpenGL frame buffer object (FBO) to manage the frame
    // buffer in an object oriented manner
    {
        GlTexture sum1(4, 4, GL_RGBA, GL_RGBA, GL_FLOAT);
        GlTexture sum2(4, 4, GL_LUMINANCE, GL_LUMINANCE32F_ARB, GL_FLOAT);
        GlTexture sum3(4, 4, GL_RED, GL_LUMINANCE32F_ARB, GL_FLOAT);
        GlTexture sum4(4, 4, GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT);
        GlTexture sum5(4, 4, GL_RED, GL_LUMINANCE, GL_FLOAT);
        GlTexture sum6(4, 4, GL_RGBA, GL_LUMINANCE32F_ARB, GL_FLOAT);
        GlTexture sum7(4, 4, GL_RGBA, GL_LUMINANCE, GL_FLOAT);
        GlTexture sum8(4, 4, GL_RED, GL_RGBA, GL_FLOAT);
        GlTexture sum9(4, 4, GL_LUMINANCE, GL_RGBA, GL_FLOAT);

        {GlFrameBuffer fb(&sum1);}
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(&sum2));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(&sum3));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(&sum4));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(&sum5));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(&sum6));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(&sum7));
        {GlFrameBuffer fb(&sum8);}
        {GlFrameBuffer fb(&sum9);}
    }
}
