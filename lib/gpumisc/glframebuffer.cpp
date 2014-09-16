#include "glframebuffer.h"

#include "GlException.h"
#include "gl.h"
#include "exceptionassert.h"
#include "tasktimer.h"
#include "backtrace.h"

//#define DEBUG_INFO
#define DEBUG_INFO if(0)

using boost::format;

class GlFrameBufferException: virtual public boost::exception, virtual public std::exception {};

GlFrameBuffer::
        GlFrameBuffer()
            :
              GlFrameBuffer(0,0)
{
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
        TaskInfo("GlFrameBuffer exception\n%s", boost::current_exception_diagnostic_information ().c_str());

        if (rboId_) glDeleteRenderbuffers(1, &rboId_);
        if (fboId_) glDeleteFramebuffers(1, &fboId_);

        throw;
    }
}

GlFrameBuffer::
        GlFrameBuffer(unsigned textureid, int width, int height)
    :
    fboId_(0),
    rboId_(0),
    own_texture_(0),
    textureid_(textureid),
    enable_depth_component_(false),
    texture_width_(width),
    texture_height_(height)
{
    EXCEPTION_ASSERT_LESS(0u, textureid);
    EXCEPTION_ASSERT_LESS(0, width);
    EXCEPTION_ASSERT_LESS(0, height);

    init();

    try
    {
        recreate(texture_width_, texture_height_);
    }
    catch(...)
    {
        TaskInfo("GlFrameBuffer() caught exception");
        if (rboId_) glDeleteRenderbuffers(1, &rboId_);
        if (fboId_) glDeleteFramebuffers(1, &fboId_);
        unbindFrameBuffer ();

        throw;
    }
}

GlFrameBuffer::
        GlFrameBuffer(const GlTexture& texture)
    :
      GlFrameBuffer(texture.getOpenGlTextureId (), texture.getWidth (), texture.getHeight ())
{
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
    glDeleteRenderbuffers(1, &rboId_);

    DEBUG_INFO TaskInfo("glDeleteFramebuffersEXT");
    glDeleteFramebuffers(1, &fboId_);

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
    GlException_SAFE_CALL( glGetIntegerv (GL_FRAMEBUFFER_BINDING, &prev_fbo_) );
    GlException_SAFE_CALL( glBindFramebuffer(GL_FRAMEBUFFER, fboId_));
}

void GlFrameBuffer::
        unbindFrameBuffer()
{
    GlException_SAFE_CALL( glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo_));
}


void GlFrameBuffer::
        recreate(int width, int height)
{
    const char* action = "Resizing";
    if (0==width)
    {
        action = "Creating";

        GLint viewport[4] = {0,0,0,0};
        glGetIntegerv(GL_VIEWPORT, viewport);
        width = viewport[2];
        height = viewport[3];

        GLint max_texture_size;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);

        if (width>max_texture_size || height>max_texture_size || width == 0 || height == 0)
            throw std::logic_error("Can't call GlFrameBuffer when no valid viewport is active");
    }

    if (width == texture_width_ && height == texture_height_ && (enable_depth_component_?rboId_:true) && fboId_)
        return;

    DEBUG_INFO TaskTimer tt("%s fbo(%u, %u)", action, width, height);

    // if (rboId_) { glDeleteRenderbuffers(1, &rboId_); rboId_ = 0; }
    // if (fboId_) { glDeleteFramebuffers(1, &fboId_); fboId_ = 0; }

    if (width != texture_width_ || height != texture_height_) {
        EXCEPTION_ASSERTX(own_texture_, format("glframebuffer: old(%g, %g), new(%g, %g)")
                          % texture_width_ % texture_height_ % width % height);
        own_texture_->reset(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
        texture_width_ = width;
        texture_height_ = height;
    }

    if (enable_depth_component_) {
        if (!rboId_)
            GlException_SAFE_CALL( glGenRenderbuffers(1, &rboId_) );

        GlException_SAFE_CALL( glBindRenderbuffer(GL_RENDERBUFFER, rboId_) );
        GlException_SAFE_CALL( glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height) );
        GlException_SAFE_CALL( glBindRenderbuffer(GL_RENDERBUFFER, 0) );
    }

    {
        if (!fboId_)
            GlException_SAFE_CALL( glGenFramebuffers(1, &fboId_) );

        auto fbo_raii = getScopeBinding();

        GlException_SAFE_CALL( glFramebufferTexture2D(
                                  GL_FRAMEBUFFER,
                                  GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_2D,
                                  textureid_,
                                  0) );

        if (enable_depth_component_)
            GlException_SAFE_CALL( glFramebufferRenderbuffer(
                                         GL_FRAMEBUFFER,
                                         GL_DEPTH_ATTACHMENT,
                                         GL_RENDERBUFFER,
                                         rboId_));

#ifdef _DEBUG
        int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

        if (GL_FRAMEBUFFER_UNSUPPORTED == status)
          {
            BOOST_THROW_EXCEPTION(GlFrameBufferException() << errinfo_format(boost::format(
                    "Got GL_FRAMEBUFFER_UNSUPPORTED. See GlFrameBuffer::test for supported formats")) << Backtrace::make ());
          }

        EXCEPTION_ASSERT_EQUALS( GL_FRAMEBUFFER_COMPLETE, status );
#endif
    }

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

        if (!glewIsSupported( "GL_framebuffer_object" )) {
            BOOST_THROW_EXCEPTION(GlFrameBufferException() << errinfo_format(boost::format(
                    "Failed to get minimal extensions\n"
                    "Sonic AWE requires:\n"
                    "  GL_framebuffer_object\n")) << Backtrace::make ());
        }
    }
#endif
}


//////// test //////

#include "expectexception.h"
#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget

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
#ifndef GL_ES_VERSION_2_0 // OpenGL ES doesn't have GL_LUMINANCE32F_ARB
        GlTexture sum2(4, 4, GL_LUMINANCE, GL_LUMINANCE32F_ARB, GL_FLOAT);
        GlTexture sum3(4, 4, GL_RED, GL_LUMINANCE32F_ARB, GL_FLOAT);
#endif
        GlTexture sum4(4, 4, GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT);
        GlTexture sum5(4, 4, GL_RED, GL_LUMINANCE, GL_FLOAT);
#ifndef GL_ES_VERSION_2_0 // OpenGL ES doesn't have GL_LUMINANCE32F_ARB
        GlTexture sum6(4, 4, GL_RGBA, GL_LUMINANCE32F_ARB, GL_FLOAT);
#endif
        GlTexture sum7(4, 4, GL_RGBA, GL_LUMINANCE, GL_FLOAT);
        GlTexture sum8(4, 4, GL_RED, GL_RGBA, GL_FLOAT);
        GlTexture sum9(4, 4, GL_LUMINANCE, GL_RGBA, GL_FLOAT);

        {GlFrameBuffer fb(sum1.getOpenGlTextureId (), 4,4);}
#ifdef _DEBUG
#ifndef GL_ES_VERSION_2_0 // OpenGL ES doesn't have GL_LUMINANCE32F_ARB
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum2.getOpenGlTextureId (), 4,4));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum3.getOpenGlTextureId (), 4,4));
#endif
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum4.getOpenGlTextureId (), 4,4));
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum5.getOpenGlTextureId (), 4,4));
#ifndef GL_ES_VERSION_2_0 // OpenGL ES doesn't have GL_LUMINANCE32F_ARB
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum6.getOpenGlTextureId (), 4,4));
#endif
        EXPECT_EXCEPTION(GlFrameBufferException, GlFrameBuffer fb(sum7.getOpenGlTextureId (), 4,4));
#endif
        {GlFrameBuffer fb(sum8.getOpenGlTextureId (), 4,4);}
        {GlFrameBuffer fb(sum9.getOpenGlTextureId (), 4,4);}
    }
}
