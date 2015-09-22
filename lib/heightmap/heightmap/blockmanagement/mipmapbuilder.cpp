#include "mipmapbuilder.h"
#include "exceptionassert.h"
#include "log.h"
#include "glstate.h"
#include "gluperspective.h"
#include "GlException.h"

namespace Heightmap {
namespace BlockManagement {

MipmapBuilder::
        MipmapBuilder()
    :
      fbo_(0),
      vbo_(0)
{

}


MipmapBuilder::
        ~MipmapBuilder()
{
    if ((vbo_ || fbo_) && !QOpenGLContext::currentContext ()) {
        Log ("%s: destruction without gl context leaks fbo %d and vbo %d") % __FILE__ % fbo_ % vbo_;
        return;
    }

    if (vbo_) GlState::glDeleteBuffers (1, &vbo_);
    vbo_ = 0;

    if (fbo_) glDeleteFramebuffers(1, &fbo_);
    fbo_ = 0;
}


void MipmapBuilder::
        init()
{
    if (vbo_)
        return;

    EXCEPTION_ASSERT(QOpenGLContext::currentContext ());

    for (int i=0; i<MipmapOperator_Last; i++)
    {
        ShaderInfo info;
        switch(i) {
        case MipmapOperator_ArithmeticMean:
            info.p = ShaderResource::loadGLSLProgram (":/shaders/mipmapbuilder.vert",":/shaders/mipmapbuilder.frag",0,"#define MipmapOperator_ArithmeticMean");
            break;
        case MipmapOperator_GeometricMean:
            info.p = ShaderResource::loadGLSLProgram (":/shaders/mipmapbuilder.vert",":/shaders/mipmapbuilder.frag",0,"#define MipmapOperator_GeometricMean");
            break;
        case MipmapOperator_HarmonicMean:
            info.p = ShaderResource::loadGLSLProgram (":/shaders/mipmapbuilder.vert",":/shaders/mipmapbuilder.frag",0,"#define MipmapOperator_HarmonicMean");
            break;
        case MipmapOperator_SqrMean:
            info.p = ShaderResource::loadGLSLProgram (":/shaders/mipmapbuilder.vert",":/shaders/mipmapbuilder.frag",0,"#define MipmapOperator_SqrMean");
            break;
        case MipmapOperator_Max:
            info.p = ShaderResource::loadGLSLProgram (":/shaders/mipmapbuilder.vert",":/shaders/mipmapbuilder.frag",0,"#define MipmapOperator_Max");
            break;
        case MipmapOperator_Min:
            info.p = ShaderResource::loadGLSLProgram (":/shaders/mipmapbuilder.vert",":/shaders/mipmapbuilder.frag",0,"#define MipmapOperator_Min");
            break;
        case MipmapOperator_OTA:
            info.p = ShaderResource::loadGLSLProgram (":/shaders/mipmapbuilder.vert",":/shaders/mipmapbuilder.frag",0,"#define MipmapOperator_OTA");
            break;
        default:
            EXCEPTION_ASSERTX(i,"Unknown MipmapOperator");
        }

        if (info.p->isLinked ())
        {
            int program = info.p->programId();
            info.qt_Vertex = glGetAttribLocation (program, "qt_Vertex");
            info.qt_MultiTexCoord0 = glGetAttribLocation (program, "qt_MultiTexCoord0");
            info.qt_Texture0 = glGetUniformLocation(program, "qt_Texture0");
            info.subtexeloffset = glGetUniformLocation(program, "subtexeloffset");
            info.level = glGetUniformLocation(program, "level");

            GlState::glUseProgram (info.p->programId());
            glUniform1i(info.qt_Texture0, 0); // GL_TEXTURE0 + i
        }

        shaders_[i] = std::move(info);
    }

    glGenFramebuffers(1, &fbo_);
    glGenBuffers (1, &vbo_);
    float vertices[] = {
        -1, -1, 0, 0,
         1, -1, 1, 0,
        -1,  1, 0, 1,
         1,  1, 1, 1,
    };

    GlState::glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}


void MipmapBuilder::
        buildMipmaps(const GlTexture& tex, MipmapOperator op, int maxlevels)
{
    // assume mipmaps are allocated in tex
    if (op >= MipmapOperator_Last)
        EXCEPTION_ASSERTX(op, "Unknown MipmapOperator");

    EXCEPTION_ASSERT_EQUALS((int)tex.getMinFilter (), GL_LINEAR_MIPMAP_LINEAR);

    init();
    ShaderInfo& info = shaders_[op];
    if (!info.p->isLinked ())
        return;

    GlState::glUseProgram (info.p->programId());

    GlState::glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    GlState::glEnableVertexAttribArray (info.qt_Vertex);
    GlState::glEnableVertexAttribArray (info.qt_MultiTexCoord0);

    struct vertex_format {
        float x, y, u, v;
    };
    glVertexAttribPointer (info.qt_Vertex, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), 0);
    glVertexAttribPointer (info.qt_MultiTexCoord0, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), (float*)0 + 2);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_);
    glBindTexture( GL_TEXTURE_2D, tex.getOpenGlTextureId ());

     // the results are the same with LINEAR_MIPMAP_LINEAR as well but this ensures that no unnecessary texture reads are performed
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST );

    GlState::glDisable (GL_DEPTH_TEST);
    GlState::glDisable (GL_CULL_FACE);
    GlState::glDisable (GL_BLEND);

    if (maxlevels<0)
        maxlevels = tex.getWidth (); // whatever is bigger than log2(min(tex.getWidth (),tex.getHeight ()))

    for (int level=1; level<maxlevels; level++)
    {
        int w = tex.getWidth () >> level;
        int h = tex.getHeight () >> level;

        if (w == 0 || h == 0)
            break;

        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, tex.getOpenGlTextureId (), level);

        glClearColor (0.01+0.01*level,0,0,0);
        glClear (GL_COLOR_BUFFER_BIT);

        glUniform2f(info.subtexeloffset, 0.25/w, 0.25/h);
        glUniform1f(info.level, level-1);
        glViewport(0, 0, w, h);
        GlState::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }

    GlState::glEnable (GL_DEPTH_TEST);
    GlState::glEnable (GL_CULL_FACE);

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, QOpenGLContext::currentContext ()->defaultFramebufferObject ());

    GlState::glDisableVertexAttribArray (info.qt_MultiTexCoord0);
    GlState::glDisableVertexAttribArray (info.qt_Vertex);
    GlState::glBindBuffer(GL_ARRAY_BUFFER, 0);
}


} // namespace BlockManagement
} // namespace Heightmap

#include "log.h"
#include "cpumemorystorage.h"
#include "heightmap/render/blocktextures.h"
#include "gltextureread.h"
#include "datastoragestring.h"

#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget

namespace Heightmap {
namespace BlockManagement {

void MipmapBuilder::
        test()
{
    std::string name = "MipmapBuilder";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
#ifndef LEGACY_OPENGL
    QGLFormat f = QGLFormat::defaultFormat ();
    f.setProfile( QGLFormat::CoreProfile );
    f.setVersion( 3, 2 );
    QGLFormat::setDefaultFormat (f);
#endif
    QGLWidget w;
    w.makeCurrent ();
#ifndef LEGACY_OPENGL
    GLuint VertexArrayID;
    GlException_SAFE_CALL( glGenVertexArrays(1, &VertexArrayID) );
    GlException_SAFE_CALL( glBindVertexArray(VertexArrayID) );
#endif
    GlState::assume_default_gl_states ();

    // It should build custom mipmaps fast
    {
        const int levels = 3;
        BlockLayout bl(4,4,4,levels);
        Render::BlockTextures::Scoped bt_raii(bl.texels_per_row (), bl.texels_per_column ());

        GlTexture::ptr tex = Render::BlockTextures::get1 ();
        tex->bindTexture ();
        glGenerateMipmap (GL_TEXTURE_2D);
        tex->setMinFilter (GL_LINEAR_MIPMAP_LINEAR);
        DataStorage<float>::ptr data0, level1, level2;

        {
            float srcdata[]={ 1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 10, 11, 12,
                              13, 14, 15, 16};

            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            GlException_SAFE_CALL( MipmapBuilder().buildMipmaps (*tex, MipmapOperator_ArithmeticMean) );

            float expected1[]={ 14/4., 22/4.,
                                46/4., 54/4. };
            float expected2[]={ 8.5 };
            data0 = GlTextureRead(*tex).readFloat (0, GL_RED);
            level1 = GlTextureRead(*tex).readFloat (1, GL_RED);
            level2 = GlTextureRead(*tex).readFloat (2, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(srcdata, sizeof(srcdata), data0);
            COMPARE_DATASTORAGE(expected1, sizeof(expected1), level1);
            COMPARE_DATASTORAGE(expected2, sizeof(expected2), level2);

            tex->bindTexture ();

            // it should be identical to glGenerateMipmap
            glGenerateMipmap (GL_TEXTURE_2D);
            data0 = GlTextureRead(*tex).readFloat (0, GL_RED);
            level1 = GlTextureRead(*tex).readFloat (1, GL_RED);
            level2 = GlTextureRead(*tex).readFloat (2, GL_RED);
            COMPARE_DATASTORAGE(srcdata, sizeof(srcdata), data0);
            COMPARE_DATASTORAGE(expected1, sizeof(expected1), level1);
            COMPARE_DATASTORAGE(expected2, sizeof(expected2), level2);
        }

        {
            float srcdata[]={ 1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 10, 11, 12,
                              13, 14, 15, 16};

            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            GlException_SAFE_CALL( MipmapBuilder().buildMipmaps (*tex, MipmapOperator_GeometricMean) );

            float expected1[]={ 2.78125, 5.0898438147,
                                11.3125, 13.3359371933 };
            float sum = 0;
            for (int i=0; i<16; i++) sum += std::log(srcdata[i]);
            float expected2[]={ std::exp(sum/16.f)-0.00359154f }; // error due to mediump. 6.79687523163
            data0 = GlTextureRead(*tex).readFloat (0, GL_RED);
            level1 = GlTextureRead(*tex).readFloat (1, GL_RED);
            level2 = GlTextureRead(*tex).readFloat (2, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(srcdata, sizeof(srcdata), data0);
            COMPARE_DATASTORAGE(expected1, sizeof(expected1), level1);
            COMPARE_DATASTORAGE(expected2, sizeof(expected2), level2);
        }

        {
            float srcdata[]={ 1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 10, 11, 12,
                              13, 14, 15, 16};

            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            GlException_SAFE_CALL( MipmapBuilder().buildMipmaps (*tex, MipmapOperator_HarmonicMean) );

            float expected1[]={ 2.14257809265, 4.69921856949,
                                11.125, 13.1796876022 };
            float expected2[]={ 4.73046874735 };
            data0 = GlTextureRead(*tex).readFloat (0, GL_RED);
            level1 = GlTextureRead(*tex).readFloat (1, GL_RED);
            level2 = GlTextureRead(*tex).readFloat (2, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(srcdata, sizeof(srcdata), data0);
            COMPARE_DATASTORAGE(expected1, sizeof(expected1), level1);
            COMPARE_DATASTORAGE(expected2, sizeof(expected2), level2);
        }


        {
            float srcdata[]={ 1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 10, 11, 12,
                              13, 14, 15, 16};

            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            GlException_SAFE_CALL( MipmapBuilder().buildMipmaps (*tex, MipmapOperator_SqrMean) );

            float expected1[]={ 4.0585938147, 5.8710938147,
                                11.6796876022, 13.6562495911 };
            float sum = 0;
            for (int i=0; i<16; i++) sum +=srcdata[i]*srcdata[i];
            float expected2[]={ std::sqrtf(sum/16.f) - 0.00547695f }; // error due to mediump. 9.66406286102
            data0 = GlTextureRead(*tex).readFloat (0, GL_RED);
            level1 = GlTextureRead(*tex).readFloat (1, GL_RED);
            level2 = GlTextureRead(*tex).readFloat (2, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(srcdata, sizeof(srcdata), data0);
            COMPARE_DATASTORAGE(expected1, sizeof(expected1), level1);
            COMPARE_DATASTORAGE(expected2, sizeof(expected2), level2);
        }

        {
            float srcdata[]={ 1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 10, 11, 12,
                              13, 14, 15, 16};

            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            GlException_SAFE_CALL( MipmapBuilder().buildMipmaps (*tex, MipmapOperator_Max) );

            float expected1[]={ 6,  8,
                                 14, 16 };
            float expected2[]={ 16 };
            data0 = GlTextureRead(*tex).readFloat (0, GL_RED);
            level1 = GlTextureRead(*tex).readFloat (1, GL_RED);
            level2 = GlTextureRead(*tex).readFloat (2, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(srcdata, sizeof(srcdata), data0);
            COMPARE_DATASTORAGE(expected1, sizeof(expected1), level1);
            COMPARE_DATASTORAGE(expected2, sizeof(expected2), level2);
        }

        {
            float srcdata[]={ 16, 15, 14, 13,
                              12, 11, 10, 9,
                              8,  7,  6,  5,
                              4,  3,  2,  1};

            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            GlException_SAFE_CALL( MipmapBuilder().buildMipmaps (*tex, MipmapOperator_Max) );

            float expected1[]={ 16, 14,
                                 8,  6 };
            float expected2[]={ 16 };
            data0 = GlTextureRead(*tex).readFloat (0, GL_RED);
            level1 = GlTextureRead(*tex).readFloat (1, GL_RED);
            level2 = GlTextureRead(*tex).readFloat (2, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(srcdata, sizeof(srcdata), data0);
            COMPARE_DATASTORAGE(expected1, sizeof(expected1), level1);
            COMPARE_DATASTORAGE(expected2, sizeof(expected2), level2);
        }

        {
            float srcdata[]={ 1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 10, 11, 12,
                              13, 14, 15, 16};

            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            GlException_SAFE_CALL( MipmapBuilder().buildMipmaps (*tex, MipmapOperator_Min) );

            float expected1[]={ 1,  3,
                                 9, 11 };
            float expected2[]={ 1 };
            data0 = GlTextureRead(*tex).readFloat (0, GL_RED);
            level1 = GlTextureRead(*tex).readFloat (1, GL_RED);
            level2 = GlTextureRead(*tex).readFloat (2, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(srcdata, sizeof(srcdata), data0);
            COMPARE_DATASTORAGE(expected1, sizeof(expected1), level1);
            COMPARE_DATASTORAGE(expected2, sizeof(expected2), level2);
        }

        {
            float srcdata[]={ -198, 2, -3430, 8528,
                              5, 614, 7, 4,
                              10, -9136, -11408, 12,
                              13, 14128, 16496, 16};

            tex->bindTexture ();
            GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, 4, 4, GL_RED, GL_FLOAT, srcdata) );

            GlException_SAFE_CALL( MipmapBuilder().buildMipmaps (*tex, MipmapOperator_OTA) );

            float expected1[]={ (2+5)/2.,   (7+4)/2.,
                                (10+13)/2., (12+16)/2. };
            float expected2[]={ ((10+13)/2. + (7+4)/2.)/2. };
            data0 = GlTextureRead(*tex).readFloat (0, GL_RED);
            level1 = GlTextureRead(*tex).readFloat (1, GL_RED);
            level2 = GlTextureRead(*tex).readFloat (2, GL_RED);

            //DataStorage<float>::ptr expectedptr = CpuMemoryStorage::BorrowPtr(DataStorageSize(4,4), expected);
            //PRINT_DATASTORAGE(expectedptr, "");
            //PRINT_DATASTORAGE(data, "");

            COMPARE_DATASTORAGE(srcdata, sizeof(srcdata), data0);
            COMPARE_DATASTORAGE(expected1, sizeof(expected1), level1);
            COMPARE_DATASTORAGE(expected2, sizeof(expected2), level2);
        }
    }
}

} // namespace BlockManagement
} // namespace Heightmap
