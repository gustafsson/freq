#include "gltextureread.h"
#include "gl.h"
#include "GlTexture.h"
#include "log.h"
#include "GlException.h"

#include <QApplication>
#include <QGLWidget>

// https://www.opengl.org/sdk/docs/man/xhtml/glGetTexImage.xml


GlTextureRead::
        GlTextureRead(int texture)
    :
      texture(texture)
{
}


DataStorage<float>::Ptr GlTextureRead::
        read(int level, int format)
{
    // assumes GL_PIXEL_PACK_BUFFER_BINDING is 0
    GLint width=0, height=0, pack_alignment=0;

    GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, texture) );
    GlException_SAFE_CALL( glGetTexLevelParameteriv (GL_TEXTURE_2D, level, GL_TEXTURE_WIDTH, &width) );
    GlException_SAFE_CALL( glGetTexLevelParameteriv (GL_TEXTURE_2D, level, GL_TEXTURE_HEIGHT, &height) );
    GlException_SAFE_CALL( glGetIntegerv (GL_PACK_ALIGNMENT, &pack_alignment) );
    GlException_SAFE_CALL( glPixelStorei (GL_PACK_ALIGNMENT, 4) );

    EXCEPTION_ASSERT_LESS( 0, width );
    EXCEPTION_ASSERT_LESS( 0, height );

    DataStorage<float>::Ptr data(new DataStorage<float>(width, height));

    GlException_SAFE_CALL( glGetTexImage(GL_TEXTURE_2D, level, format, GL_FLOAT,  data->getCpuMemory()) );

    // restore
    GlException_SAFE_CALL( glPixelStorei (GL_PACK_ALIGNMENT, pack_alignment) );
    GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, 0) );

    return data;
}


void GlTextureRead::
        test()
{
    int argc = 0;
    char* argv = 0;
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should read the contents of an OpenGL texture.
    {
        // There must be a current OpenGL context
        EXCEPTION_ASSERT(QGLContext::currentContext ());

        const char* extensions = (const char*)glGetString(GL_EXTENSIONS);
        EXCEPTION_ASSERT(extensions);

        bool hasTextureFloat = 0 != strstr( extensions, "GL_ARB_texture_float" );
        EXCEPTION_ASSERT(hasTextureFloat);

        float srcdata[]={1,2,3,4,
                     2,3,4,5,
                     3,4,5,6,
                     4,5,6,7};
        GlTexture src(4, 4, GL_LUMINANCE, GL_LUMINANCE32F_ARB, GL_FLOAT, srcdata);

        DataStorage<float>::Ptr data = GlTextureRead(src.getOpenGlTextureId ()).read (0,GL_RED);
        float *p = data->getCpuMemory ();

        EXCEPTION_ASSERT_EQUALS(0, memcmp(p, srcdata, sizeof(srcdata)));
    }
}
