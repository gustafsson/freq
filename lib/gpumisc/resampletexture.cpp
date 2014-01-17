#include "resampletexture.h"
#include "gl.h"
#include "glPushContext.h"

#include "exceptionassert.h"
#include "log.h"
#include "gltextureread.h"
#include "GlException.h"

#include <QApplication>
#include <QGLWidget>

ResampleTexture::Area::
        Area(float x1, float y1, float x2, float y2)
    :
      x1(x1), y1(y1),
      x2(x2), y2(y2)
{}


ResampleTexture::
        ResampleTexture(GlTexture* dest, Area destarea)
    :
      fbo(dest),
      dest(dest),
      destarea(destarea)
{
}


void ResampleTexture::
        clear(float r, float g, float b, float a)
{
    GlException_SAFE_CALL( glClearColor (r,g,b,a) );
    GlFrameBuffer::ScopeBinding fboBinding = fbo.getScopeBinding();
    GlException_SAFE_CALL( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );
}


void ResampleTexture::
        operator ()(GlTexture* source, Area area)
{
    GlFrameBuffer::ScopeBinding fboBinding = fbo.getScopeBinding();
    GlException_SAFE_CALL( glViewport(0, 0, dest->getWidth (), dest->getHeight ()) );

    glPushMatrixContext mpc( GL_PROJECTION );
    glLoadIdentity();
    glOrtho(destarea.x1, destarea.x2, destarea.y1, destarea.y2, -10,10);
    glPushMatrixContext mc( GL_MODELVIEW );
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);

    {
        GlTexture::ScopeBinding texObjBinding = source->getScopeBinding();
        glBegin(GL_TRIANGLE_STRIP);
            glTexCoord2f(0,0); glVertex2f(area.x1,area.y1);
            glTexCoord2f(1,0); glVertex2f(area.x2,area.y1);
            glTexCoord2f(0,1); glVertex2f(area.x1,area.y2);
            glTexCoord2f(1,1); glVertex2f(area.x2,area.y2);
        glEnd();
        // avoid glBegin/glEnd -> glDrawElements(GL_TRIANGLE_STRIP, vbo_size, BLOCK_INDEX_TYPE, 0);
    }
    glEnable(GL_DEPTH_TEST);
}


/////////////////// tests ////////////////////////


static void compare(float* expected, size_t sizeof_expected, DataStorage<float>::Ptr data)
{
    EXCEPTION_ASSERT(data);
    EXCEPTION_ASSERT_EQUALS(sizeof_expected, data->numberOfBytes ());

    float *p = data->getCpuMemory ();

    if (0 != memcmp(p, expected, sizeof_expected))
    {
        Log("%s") % data->numberOfElements ();
        for (size_t i=0; i<data->numberOfElements (); i++)
            Log("%s: %s\t%s\t%s") % i % p[i] % expected[i] % (p[i] - expected[i]);

        EXCEPTION_ASSERT_EQUALS(0, memcmp(p, expected, sizeof_expected));
    }
}


void ResampleTexture::
        test()
{
    int argc = 0;
    char* argv = 0;
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should paint a texture on top of another texture.
    {
        // There must be a current OpenGL context
        EXCEPTION_ASSERT(QGLContext::currentContext ());

        const char* extensions = (const char*)glGetString(GL_EXTENSIONS);
        EXCEPTION_ASSERT(extensions);

        bool hasTextureFloat = 0 != strstr( extensions, "GL_ARB_texture_float" );
        EXCEPTION_ASSERT(hasTextureFloat);

        float a = 0.99215691f;
        float b = 0.49803924f;
        float srcdata[]={ 1, 0, 0, .5,
                          0, 0, 0, 0,
                          0, 0, 0, 0,
                         .5, 0, 0, .5};
        float expected1[]={0, 0, 0, 0,
                           0, a, b, 0,
                           0, b, b, 0,
                           0, 0, 0, 0};
        float expected2[]={1, 0, 0, b,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           b, 0, 0, b};
        float expected3[]={1, 0, 0, b,
                           0, a, b, 0,
                           0, b, b, 0,
                           b, 0, 0, b};
        float expected4[]={1, 0, 0, b,
                           0, 1, 0, 0,
                           0, 0, 0, 0,
                           b, 0, 0, b};

        GlTexture dest(4, 4);
        GlTexture src(4, 4, GL_LUMINANCE, GL_LUMINANCE32F_ARB, GL_FLOAT, srcdata);

        ResampleTexture rt(&dest,Area(0,0,3,3));

        rt.clear ();
        rt(&src,Area(1,1,2,2));

        DataStorage<float>::Ptr data = GlTextureRead(dest.getOpenGlTextureId ()).readFloat (0, GL_RED);

        compare(expected1, sizeof(expected1), data);

        rt(&src,Area(0,0,3,3));
        data = GlTextureRead(dest.getOpenGlTextureId ()).readFloat (0, GL_RED);
        compare(expected2, sizeof(expected2), data);

        rt(&src,Area(1,1,2,2));
        data = GlTextureRead(dest.getOpenGlTextureId ()).readFloat (0, GL_RED);
        compare(expected3, sizeof(expected3), data);

        rt(&src,Area(1,1,2.5,2.5));
        data = GlTextureRead(dest.getOpenGlTextureId ()).readFloat (0, GL_RED);
        compare(expected4, sizeof(expected4), data);
    }
}
