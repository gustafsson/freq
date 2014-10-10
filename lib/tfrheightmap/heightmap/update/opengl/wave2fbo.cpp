#include "wave2fbo.h"
#include "gl.h"
#include "GlException.h"
#include "cpumemorystorage.h"
#include "tasktimer.h"
#include "log.h"

#include <QOpenGLShaderProgram>

namespace Heightmap {
namespace Update {
namespace OpenGL {

Wave2Fbo::
        Wave2Fbo(Signal::pMonoBuffer b)
    :
      b_(b),
      N(b_->number_of_samples ())
{
    struct vertex_format_xy {
        float x, y;
    };

    float t0 = b_->start ();
    float ifs = 1.0f/b_->sample_rate ();
    float t1 = t0 + N*ifs;

    GlException_CHECK_ERROR();

    glGenBuffers (1, &vbo_); // Generate 1 buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData (GL_ARRAY_BUFFER, sizeof(vertex_format_xy)*(N + 4), 0, GL_STATIC_DRAW);
#ifndef GL_ES_VERSION_2_0
    vertex_format_xy* d = (vertex_format_xy*)glMapBuffer (GL_ARRAY_BUFFER, GL_WRITE_ONLY);
#else
    vertex_format_xy* d = (vertex_format_xy*)glMapBufferOES (GL_ARRAY_BUFFER, GL_WRITE_ONLY_OES);
#endif
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    float* p = CpuMemoryStorage::ReadOnly<1>(b_->waveform_data()).ptr ();
    for (int i=0; i<N; i++)
        d[i] = vertex_format_xy{ t0 + i*ifs, 0.5f + 0.5f*p[i] };

    d[N+0] = vertex_format_xy{ t0, 0 };
    d[N+1] = vertex_format_xy{ t1, 0 };
    d[N+2] = vertex_format_xy{ t0, 1 };
    d[N+3] = vertex_format_xy{ t1, 1 };

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
#ifndef GL_ES_VERSION_2_0
    glUnmapBuffer (GL_ARRAY_BUFFER);
#else
    glUnmapBufferOES (GL_ARRAY_BUFFER);
#endif
    d = 0;
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GlException_CHECK_ERROR();
}


Wave2Fbo::
        ~Wave2Fbo()
{
    if (vbo_)
        glDeleteBuffers (1, &vbo_);
}


void Wave2Fbo::
        draw(const glProjection& p)
{
    if (!m_program) {
        m_program = new QOpenGLShaderProgram();
        m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                           "attribute highp vec4 vertices;"
                                           "uniform highp mat4 ModelViewProjectionMatrix;"
                                           "void main() {"
                                           "    gl_Position = ModelViewProjectionMatrix*vertices;"
                                           "}");
        m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                           "uniform lowp vec4 rgba;"
                                           "void main() {"
                                           "    gl_FragColor = rgba;"
                                           "}");

        m_program->bindAttributeLocation("vertices", 0);
        if (!m_program->link())
            Log("wave2fbo: invalid shader\n%s")
                    % m_program->log ().toStdString ();
    }

    if (!m_program->isLinked ())
        return;

    m_program->bind();

    m_program->enableAttributeArray(0);

    matrixd modelview = p.modelview;
    QMatrix4x4 modelviewprojection {GLmatrixf(p.projection*modelview).transpose ().v ()};
    m_program->setUniformValue("ModelViewProjectionMatrix", modelviewprojection);
    m_program->setUniformValue("rgba", QVector4D(0.0,0.0,0.0,0.5));

    //int uniModelViewProjectionMatrix = glGetUniformLocation (program_, "qt_ModelViewProjectionMatrix");
    //glUniformMatrix4fv (uniModelViewProjectionMatrix, 1, false, M.projection * M.modelview);

    //TaskTimer tt(boost::format("Wave2Fbo::draw %s") % b_->getInterval ());

    GlException_CHECK_ERROR();
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // Draw clear rectangle
    m_program->setUniformValue("rgba", QVector4D(0.0,0.0,0.0,1.0));
    glDrawArrays(GL_TRIANGLE_STRIP, N, 4);

    // Draw waveform
    glEnable (GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//    glColor4f(.5f, 0.f, 0.f, 0.001f);
    m_program->setUniformValue("rgba", QVector4D(1.0,0.0,0.0,1.0));
    glDrawArrays(GL_LINE_STRIP, 0, N);
    glDisable (GL_BLEND);

    m_program->disableAttributeArray (0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GlException_CHECK_ERROR();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
