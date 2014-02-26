#include "glframebuffer.h"

#include "GlException.h"
#include "gl.h"
#include "exceptionassert.h"
#include "tasktimer.h"
#include "backtrace.h"

//#define DEBUG_INFO
#define DEBUG_INFO if(0)

class GlFrameBufferException: virtual public boost::exception, virtual public std::exception {};

GlFrameBuffer::
        GlFrameBuffer()
            :
            fboId_(0),
            rboId_(0),
            own_texture_(new GlTexture),
            textureid_(own_texture_->getOpenGlTextureId ()),
            enable_depth_component_(true)
{
    init();

    try
    {
        recreate(0, 0);
    }
    catch(...)
    {
        if (rboId_) glDeleteRenderbuffersEXT(1, &rboId_);
        if (fboId_) glDeleteFramebuffersEXT(1, &fboId_);

        throw;
    }
}


GlFrameBuffer::
        GlFrameBuffer(int width, int height)
            :
            fboId_(0),
            rboId_(0),
            own_texture_(new GlTexture),
            textureid_(own_texture_->getOpenGlTextureId ()),
            enable_depth_component_(true)
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
        GlFrameBuffer(unsigned textureid)
    :
    fboId_(0),
    rboId_(0),
    own_texture_(0),
    textureid_(textureid),
    enable_depth_component_(false)
{
    init();

    try
    {
        glBindTexture (GL_TEXTURE_2D, textureid_);
        GlException_SAFE_CALL( glGetTexLevelParameteriv (GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &texture_width_) );
        GlException_SAFE_CALL( glGetTexLevelParameteriv (GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &texture_height_) );
        glBindTexture (GL_TEXTURE_2D, 0);

        recreate(texture_width_, texture_height_);
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
        getScopeBinding()
{
    bindFrameBuffer();
    return ScopeBinding(*this, &GlFrameBuffer::unbindFrameBuffer);
}

void GlFrameBuffer::
        bindFrameBuffer()
{
    GlException_CHECK_ERROR();

    glGetIntegerv (GL_DRAW_FRAMEBUFFER_BINDING, &prev_fbo_draw_);
    glGetIntegerv (GL_READ_FRAMEBUFFER_BINDING, &prev_fbo_read_);

    GlException_SAFE_CALL( glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboId_));

    GlException_CHECK_ERROR();
}

void GlFrameBuffer::
        unbindFrameBuffer() const
{
    GlException_CHECK_ERROR();

    GlException_SAFE_CALL( glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, prev_fbo_draw_));
    GlException_SAFE_CALL( glBindFramebufferEXT(GL_READ_FRAMEBUFFER, prev_fbo_read_));

    GlException_CHECK_ERROR();
}


void GlFrameBuffer::
        recreate(int width, int height)
{
    glBindTexture (GL_TEXTURE_2D, textureid_);
    GlException_SAFE_CALL( glGetTexLevelParameteriv (GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &texture_width_) );
    GlException_SAFE_CALL( glGetTexLevelParameteriv (GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &texture_height_) );
    glBindTexture (GL_TEXTURE_2D, 0);

    const char* action = "Resizing";
    if (0==width)
    {
        action = "Creating";

        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        width = viewport[2];
        height = viewport[3];

        GLint max_texture_size;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);

        if (width>max_texture_size || height>max_texture_size || width == 0 || height == 0)
            throw std::logic_error("Can't call GlFrameBuffer when no valid viewport is active");
    }

    if (width == texture_width_ && height == texture_height_ && rboId_ && fboId_)
        return;

    DEBUG_INFO TaskTimer tt("%s fbo(%u, %u)", action, width, height);

    // if (rboId_) { glDeleteRenderbuffersEXT(1, &rboId_); rboId_ = 0; }
    // if (fboId_) { glDeleteFramebuffersEXT(1, &fboId_); fboId_ = 0; }

    if (width != texture_width_ || height != texture_height_) {
        EXCEPTION_ASSERT(own_texture_);
        own_texture_->reset(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
        texture_width_ = width;
        texture_height_ = height;
    }

    GlException_CHECK_ERROR();

    if (enable_depth_component_) {
        if (!rboId_)
            glGenRenderbuffersEXT(1, &rboId_);

        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, rboId_);
        glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, width, height);
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
    }

    {
        if (!fboId_)
            glGenFramebuffersEXT(1, &fboId_);

        bindFrameBuffer ();

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
                                  GL_COLOR_ATTACHMENT0_EXT,
                                  GL_TEXTURE_2D,
                                  textureid_,
                                  0);

        if (enable_depth_component_)
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

        unbindFrameBuffer ();
    }

    //glFlush();

    DEBUG_INFO TaskInfo("fbo = %u", fboId_ );
    DEBUG_INFO TaskInfo("rbo = %u", fboId_ );
    DEBUG_INFO TaskInfo("texture = %u", textureid_ );

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
    std::string name = "GlFrameBuffer";
    int argc = 1;
    char * argv = &name[0];
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

        {GlFrameBuffer fb(sum1.getOpenGlTextureId ());}
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum2.getOpenGlTextureId ()));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum3.getOpenGlTextureId ()));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum4.getOpenGlTextureId ()));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum5.getOpenGlTextureId ()));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum6.getOpenGlTextureId ()));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum7.getOpenGlTextureId ()));
        {GlFrameBuffer fb(sum8.getOpenGlTextureId ());}
        {GlFrameBuffer fb(sum9.getOpenGlTextureId ());}
    }
}
