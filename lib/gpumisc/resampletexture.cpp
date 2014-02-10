#include "resampletexture.h"
#include "gl.h"
#include "glPushContext.h"

#include "exceptionassert.h"
#include "log.h"
#include "TaskTimer.h"
#include "gltextureread.h"
#include "GlException.h"
#include "datastoragestring.h"

#include <QApplication>
#include <QGLWidget>

//#define PRINT_TEXTURES
#define PRINT_TEXTURES if(0)

//#define INFO
#define INFO if(0)

ResampleTexture::Area::
        Area(float x1, float y1, float x2, float y2)
    :
      x1(x1), y1(y1),
      x2(x2), y2(y2)
{}


std::ostream& operator<<(std::ostream& o, ResampleTexture::Area a)
{
    return o << "(x=" << a.x1 << ":" << a.x2 << " y=" << a.y1 << ":" << a.y2 << ")";
}


ResampleTexture::
        ResampleTexture(unsigned dest)
    :
      fbo(dest),
      destarea(0,0,0,0)
{
    glGenBuffers (1, &vbo); // Generate 1 buffer
}


ResampleTexture::
        ~ResampleTexture()
{
    glDeleteBuffers (1, &vbo);
}


GlFrameBuffer::ScopeBinding ResampleTexture::
        enable(Area destarea)
{
    this->destarea = destarea;

    return fbo.getScopeBinding ();
}


void ResampleTexture::
        clear(float r, float g, float b, float a)
{
    INFO TaskTimer tt(boost::format("clearing %s with (%g %g %g %g)") % destarea % r % g % b % a);

    float v[4];
    glGetFloatv(GL_COLOR_CLEAR_VALUE, v);
    GlException_SAFE_CALL( glClearColor (r,g,b,a) );
    GlException_SAFE_CALL( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );
    GlException_SAFE_CALL( glClearColor (v[0],v[1],v[2],v[3]) );
}


void ResampleTexture::
        operator ()(GlTexture* source, Area area)
{
    glPushAttribContext pa(GL_ENABLE_BIT);
    glDisable (GL_DEPTH_TEST);
    glDisable (GL_BLEND);
    glDisable (GL_CULL_FACE);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    GlException_SAFE_CALL( glViewport(0, 0, fbo.getWidth (), fbo.getHeight () ) );

    glPushMatrixContext mpc( GL_PROJECTION );
    glLoadIdentity();
    glOrtho(destarea.x1, destarea.x2, destarea.y1, destarea.y2, -10,10);
    glPushMatrixContext mc( GL_MODELVIEW );
    glLoadIdentity();

    //INFO Log("Painting %s onto %s") % area % destarea;
    INFO TaskTimer tt(boost::format("Painting %s onto %s") % area % destarea);

    {
        struct vertex_format {
            float x, y, u, v;
        };

        float vertices[] = {
            area.x1, area.y1, 0, 0,
            area.x1, area.y2, 0, 1,
            area.x2, area.y1, 1, 0,
            area.x2, area.y2, 1, 1,
        };

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STREAM_DRAW);

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        glVertexPointer(2, GL_FLOAT, sizeof(vertex_format), 0);
        glTexCoordPointer(2, GL_FLOAT, sizeof(vertex_format), (float*)0 + 2);

        GlTexture::ScopeBinding texObjBinding = source->getScopeBinding();
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    PRINT_TEXTURES PRINT_DATASTORAGE(GlTextureRead(fbo.getGlTexture()).readFloat (), "fbo");
    PRINT_TEXTURES PRINT_DATASTORAGE(GlTextureRead(source->getOpenGlTextureId ()).readFloat (), "source");

    GlException_SAFE_CALL( glViewport(viewport[0], viewport[1], viewport[2], viewport[3] ) );
}


void ResampleTexture::
        drawColoredArea(Area area, float r, float g, float b, float a)
{
    glPushAttribContext pa(GL_ENABLE_BIT);
    glDisable (GL_DEPTH_TEST);
    glDisable (GL_BLEND);
    glDisable (GL_CULL_FACE);

    GlException_SAFE_CALL( glViewport(0, 0, fbo.getWidth (), fbo.getHeight () ) );

    glPushMatrixContext mpc( GL_PROJECTION );
    glLoadIdentity();
    glOrtho(destarea.x1, destarea.x2, destarea.y1, destarea.y2, -10,10);
    glPushMatrixContext mc( GL_MODELVIEW );
    glLoadIdentity();

    //INFO Log("Painting %s with color (%g, %g, %g, %g) onto %s") % area % r % g % b % a % destarea;
    INFO TaskTimer tt(boost::format("Painting %s with color (%g, %g, %g, %g) onto %s") % area % r % g % b % a % destarea);

    {
        struct vertex_format {
            float x, y, r, g, b, a;
        };

        float vertices[] = {
            area.x1, area.y1, r, g, b, a,
            area.x1, area.y2, r, g, b, a,
            area.x2, area.y1, r, g, b, a,
            area.x2, area.y2, r, g, b, a
        };
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        glVertexPointer(2, GL_FLOAT, sizeof(vertex_format), 0);
        glColorPointer(4, GL_FLOAT, sizeof(vertex_format), (float*)0 + 2);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    PRINT_TEXTURES PRINT_DATASTORAGE(GlTextureRead(fbo.getGlTexture()).readFloat (), "fbo");
}


/////////////////// tests ////////////////////////


void ResampleTexture::
        test()
{
    std::string name = "ResampleTexture";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    testInContext();
}


void ResampleTexture::
        testInContext()
{
    // It should paint a texture on top of another texture. (with GL_UNSIGNED_BYTE)
    {
        // There must be a current OpenGL context
        EXCEPTION_ASSERT(QGLContext::currentContext ());

        const char* extensions = (const char*)glGetString(GL_EXTENSIONS);
        EXCEPTION_ASSERT(extensions);

        bool hasTextureFloat = 0 != strstr( extensions, "GL_ARB_texture_float" );
        EXCEPTION_ASSERT(hasTextureFloat);

        float a = 0.99215691f;
        float b = 0.49803924f;
        float srcdata[]={ 1, 0, 0, 1.5,
                          0, 0, 0, 0,
                          0, 0, 0, 0,
                         .5, 0, 0, .5};
        float expected1[]={0, 0, 0, 0,
                           0, a, 1, 0,
                           0, b, b, 0,
                           0, 0, 0, 0};
        float expected2[]={1, 0, 0, 1,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           b, 0, 0, b};
        float expected3[]={1, 0, 0, 1,
                           0, a, 1, 0,
                           0, b, b, 0,
                           b, 0, 0, b};
        float expected4[]={1, 0, 0, 1,
                           0, 1, 0, 0,
                           0, 0, 0, 0,
                           b, 0, 0, b};
        float expected5[]={1, 1, 1, 1,
                           1, 1, 1, 1,
                           1, 1, 1, 1,
                           1, 1, 1, 1};
        float expected6[]={1, 1, 1, 1,
                           1, a, 1, 1,
                           1, b, b, 1,
                           1, 1, 1, 1};
        float expected7[]={1, 1, 1, 1,
                           1, b, b, b,
                           1, b, b, b,
                           1, b, b, b};

        GlTexture dest(4, 4);
        GlTexture src(4, 4, GL_LUMINANCE, GL_LUMINANCE32F_ARB, GL_FLOAT, srcdata);
        int destid = dest.getOpenGlTextureId ();

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));

            rt.clear ();
            rt(&src,Area(1,1,2,2));
        }
        DataStorage<float>::Ptr data;
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected1, sizeof(expected1), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt(&src,Area(0,0,3,3));
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected2, sizeof(expected2), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt(&src,Area(1,1,2,2));
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected3, sizeof(expected3), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt(&src,Area(1,1,2.5,2.5));
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected4, sizeof(expected4), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt.clear (2,1,3,4);
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected5, sizeof(expected5), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt(&src,Area(1,1,2,2));
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected6, sizeof(expected6), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt.drawColoredArea (Area(1,1,3,3), 0.5);
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected7, sizeof(expected7), data);
    }

    // It should paint a texture on top of another texture. (with GL_FLOAT)
    {
        // There must be a current OpenGL context
        EXCEPTION_ASSERT(QGLContext::currentContext ());

        const char* extensions = (const char*)glGetString(GL_EXTENSIONS);
        EXCEPTION_ASSERT(extensions);

        bool hasTextureFloat = 0 != strstr( extensions, "GL_ARB_texture_float" );
        EXCEPTION_ASSERT(hasTextureFloat);

        float a = 0.99218749f;
        float b = 0.49609374f;
        float c = 1.488281192f;
        float srcdata[]={ 1, 0, 0, 1.5,
                          0, 0, 0, 0,
                          0, 0, 0, 0,
                         .5, 0, 0, .5};
        float expected1[]={0, 0, 0, 0,
                           0, a, c, 0,
                           0, b, b, 0,
                           0, 0, 0, 0};
        float expected2[]={1,  0, 0, 1.5,
                           0,  0, 0, 0,
                           0,  0, 0, 0,
                           .5, 0, 0, .5};
        float expected3[]={1,  0, 0, 1.5,
                           0,  a, c, 0,
                           0,  b, b, 0,
                           .5, 0, 0, .5};
        float expected4[]={1,  0, 0, 1.5,
                           0,  1, 0, 0,
                           0,  0, 0, 0,
                           .5, 0, 0, .5};
        float expected5[]={2, 2, 2, 2,
                           2, 2, 2, 2,
                           2, 2, 2, 2,
                           2, 2, 2, 2};
        float expected6[]={2, 2, 2, 2,
                           2, a, c, 2,
                           2, b, b, 2,
                           2, 2, 2, 2};
        float expected7[]={2, 2, 2, 2,
                           2, .5, .5, .5,
                           2, .5, .5, .5,
                           2, .5, .5, .5};

        GlTexture dest(4, 4, GL_RGBA, GL_RGBA, GL_FLOAT);
        int destid = dest.getOpenGlTextureId ();
        GlTexture src(4, 4, GL_LUMINANCE, GL_LUMINANCE32F_ARB, GL_FLOAT, srcdata);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));

            rt.clear ();
            rt(&src,Area(1,1,2,2));
        }
        DataStorage<float>::Ptr data;
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected1, sizeof(expected1), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt(&src,Area(0,0,3,3));
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected2, sizeof(expected2), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt(&src,Area(1,1,2,2));
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected3, sizeof(expected3), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt(&src,Area(1,1,2.5,2.5));
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected4, sizeof(expected4), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt.clear (2,1,3,4);
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected5, sizeof(expected5), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt(&src,Area(1,1,2,2));
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected6, sizeof(expected6), data);

        {
            ResampleTexture rt(destid);
            GlFrameBuffer::ScopeBinding sb = rt.enable (Area(0,0,3,3));
            rt.drawColoredArea (Area(1,1,3,3), 0.5);
        }
        data = GlTextureRead(destid).readFloat (0, GL_RED);
        COMPARE_DATASTORAGE(expected7, sizeof(expected7), data);
    }
}