#include "renderregion.h"

#include "unused.h"
#include "glstate.h"
#include "glPushContext.h"
#include "GlException.h"
#include "log.h"

#include <QOpenGLShaderProgram>

namespace Heightmap {
namespace Render {

RenderRegion::
        RenderRegion()
    :
      vbo_(0)
{
}


RenderRegion::
        ~RenderRegion()
{
    if (!QOpenGLContext::currentContext ()) {
        Log ("%s: destruction without gl context leaks vbos %d and %d") % __FILE__ % vbo_;
        return;
    }

    if (vbo_)
        glDeleteBuffers (1,&vbo_);
}


void RenderRegion::
        render(const glProjecter& gl_projecter, Region r, bool drawcross)
{
    GlException_CHECK_ERROR();

    float y = 0.5f;

    if (!program_) {
        program_ = ShaderResource::loadGLSLProgramSource(
                                           "attribute highp vec4 vertices;"
                                           "uniform highp mat4 modelviewproj;"
                                           "void main() {"
                                           "    gl_Position = modelviewproj*vertices;"
                                           "}",

                                           "uniform lowp vec4 color;"
                                           "void main() {"
                                           "    gl_FragColor = color;"
                                           "}");

        static float values[] = {
            // cross_values
            0, 0, 0,
            1, 0, 1,
            1, 0, 0,
            0, 0, 1,
            0, 0, 0,
            1, 0, 0,
            1, 0, 1,
            0, 0, 1,

            // nocross_values
            0, 0, 0,
            1, 0, 0,
            1, 0, 1,
            0, 0, 1,
            0, 0, 0
        };

        glGenBuffers (1,&vbo_);
        GlState::glBindBuffer (GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(tvector<3,GLfloat>)*13, values, GL_STATIC_DRAW);
        GlState::glUseProgram (program_->programId());
    }

    if (!program_->isLinked ())
        return;

    GlState::glUseProgram (program_->programId());

    GlState::glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    GlState::glEnable(GL_BLEND);
    GlState::glEnableVertexAttribArray (0);
    glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 );

    glProjecter proj = gl_projecter;
    proj.translate (vectord(r.a.time, 0, r.a.scale));
    proj.scale (vectord(r.time(), 1, r.scale()));

    program_->setUniformValue("color", 0.8, 0.2, 0.2, 0.5);
    program_->setUniformValue("modelviewproj",
                              QMatrix4x4(GLmatrixf(proj.mvp ()).transpose ().v ()));
    int value_offs = drawcross ? 0 : 8;
    int n_values = drawcross ? 8 : 5;
    GlState::glDrawArrays(GL_LINE_STRIP, value_offs, n_values);

    proj.translate (vectord(0, y, 0));
    program_->setUniformValue("color", 0.2, 0.8, 0.8, 0.5);
    program_->setUniformValue("modelviewproj",
                              QMatrix4x4(GLmatrixf(proj.mvp()).transpose ().v ()));
    GlState::glDrawArrays(GL_LINE_STRIP, value_offs, n_values);

    GlState::glDisableVertexAttribArray (0);
    GlState::glDisable(GL_BLEND);
    GlState::glUseProgram (0);
    GlState::glBindBuffer (GL_ARRAY_BUFFER, 0);

    GlException_CHECK_ERROR();
}

} // namespace Render
} // namespace Heightmap
