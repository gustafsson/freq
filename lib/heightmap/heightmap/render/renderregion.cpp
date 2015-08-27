#include "renderregion.h"

#include "unused.h"
#include "glstate.h"
#include "glPushContext.h"
#include "GlException.h"

#include <QOpenGLShaderProgram>

namespace Heightmap {
namespace Render {

RenderRegion::RenderRegion(const glProjecter& gl_projecter)
    :
      gl_projecter_(gl_projecter)
{
}


void RenderRegion::
        render(Region r, bool drawcross)
{
    // if (!renderBlock(...) && (0 == "render red warning cross" || render_settings->y_scale < yscalelimit))
    //float y = _frustum_clip.projectionPlane[1]*.05;
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
    }

    if (!program_->isLinked ())
        return;

    program_->bind();

    GlState::glEnableVertexAttribArray (0);

    float cross_values[] = {
        0, 0, 0,
        1, 0, 1,
        1, 0, 0,
        0, 0, 1,
        0, 0, 0,
        1, 0, 0,
        1, 0, 1,
        0, 0, 1 };

    float nocross_values[] = {
        0, 0, 0,
        1, 0, 0,
        1, 0, 1,
        0, 0, 1,
        0, 0, 0
    };

    float* values = drawcross ? cross_values : nocross_values;
    int n_values = drawcross ? 8 : 5;

    program_->setAttributeArray(0, GL_FLOAT, values, 3);

    glProjecter proj = gl_projecter_;
    proj.translate (vectord(r.a.time, 0, r.a.scale));
    proj.scale (vectord(r.time(), 1, r.scale()));

    GlException_CHECK_ERROR();

    GlState::glEnable(GL_BLEND);
    glLineWidth(2);

    program_->setUniformValue("color", 0.8, 0.2, 0.2, 0.5);
    program_->setUniformValue("modelviewproj",
                              QMatrix4x4(GLmatrixf(proj.mvp ()).transpose ().v ()));
    GlState::glDrawArrays(GL_LINE_STRIP, 0, n_values);

    GlException_CHECK_ERROR();

    proj.translate (vectord(0, y, 0));
    program_->setUniformValue("color", 0.2, 0.8, 0.8, 0.5);
    program_->setUniformValue("modelviewproj",
                              QMatrix4x4(GLmatrixf(proj.mvp()).transpose ().v ()));
    GlState::glDrawArrays(GL_LINE_STRIP, 0, n_values);
    glLineWidth(1);

    GlState::glDisableVertexAttribArray (0);
    GlState::glDisable(GL_BLEND);
    program_->release();

    GlException_CHECK_ERROR();
}

} // namespace Render
} // namespace Heightmap
