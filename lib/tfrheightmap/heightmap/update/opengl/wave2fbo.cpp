#include "wave2fbo.h"
#include "gl.h"
#include "GlException.h"
#include "cpumemorystorage.h"
#include "tasktimer.h"
#include "log.h"
#include "heightmap/render/shaderresource.h"

#include <QOpenGLShaderProgram>

namespace Heightmap {
namespace Update {
namespace OpenGL {

Wave2Fbo::
        Wave2Fbo()
    :
      dv(128*1024) // 1 MB
{
    GlException_CHECK_ERROR();

    glGenBuffers (1, &vbo_); // Generate 1 buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);

    glBufferData (GL_ARRAY_BUFFER, sizeof(vertex_format_xy)*(dv.size ()), 0, GL_DYNAMIC_DRAW);
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
        draw(const glProjection& P, Signal::pMonoBuffer b)
{
    if (!m_program) {
        m_program = ShaderResource::loadGLSLProgramSource (
                                           R"vertexshader(
                                               attribute highp vec4 vertices;
                                               uniform highp mat4 ModelViewProjectionMatrix;
                                               void main() {
                                                   gl_Position = ModelViewProjectionMatrix*vertices;
                                               }
                                           )vertexshader",
                                           R"fragmentshader(
                                               uniform lowp vec4 rgba;
                                               void main() {
                                                   gl_FragColor = rgba*100.0;
                                               }
                                           )fragmentshader");

        m_program->bindAttributeLocation("vertices", 0);
        if (!m_program->link())
            Log("wave2fbo: invalid shader\n%s")
                    % m_program->log ().toStdString ();
    }

    if (!m_program->isLinked ())
        return;

    GlException_CHECK_ERROR();

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    m_program->bind();
    m_program->enableAttributeArray(0);

    matrixd modelview = P.modelview;
    modelview *= matrixd::translate (b->start (), 0.5, 0);
    modelview *= matrixd::scale (1.0/b->sample_rate (), 0.5, 1);
    QMatrix4x4 modelviewprojection {GLmatrixf(P.projection*modelview).transpose ().v ()};
    m_program->setUniformValue("ModelViewProjectionMatrix", modelviewprojection);

    GlException_CHECK_ERROR();
    glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // Draw clear rectangle
    m_program->setUniformValue("rgba", QVector4D(0.0,0.0,0.0,1.0));

    vertex_format_xy* d = &dv[0];
    int N = (int)dv.size ();
    int S = b->number_of_samples ();
    d[0] = vertex_format_xy{ 0, -1 };
    d[1] = vertex_format_xy{ float(S-1), -1 };
    d[2] = vertex_format_xy{ 0, 1 };
    d[3] = vertex_format_xy{ float(S-1), 1 };
    glBufferSubData (GL_ARRAY_BUFFER, 0, sizeof(vertex_format_xy)*4, d);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Draw waveform
    glDisable (GL_BLEND); // doesn't have alpha channel
    glLineWidth(1);
    m_program->setUniformValue("rgba", QVector4D(0.25,0.0,0.0,1.0));

    float* p = CpuMemoryStorage::ReadOnly<1>(b->waveform_data()).ptr ();

    for (int i=0,j; i<S;)
    {
        if (0<i)
            --i;

        for (j=0; i<S && j<N; ++i, ++j)
            d[j] = vertex_format_xy{ float(i), p[i] };

        // might cause implicit synchronization (wait for previous drawArrays
        // to finish) if it updates the actual buffer right away instead of
        // enqueing the data for update later
        glBufferSubData (GL_ARRAY_BUFFER, 0, sizeof(vertex_format_xy)*j, d);
        glDrawArrays(GL_LINE_STRIP, 0, j);
    }

    m_program->disableAttributeArray (0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GlException_CHECK_ERROR();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
