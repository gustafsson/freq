#include "renderregion.h"

#include "unused.h"
#include "gl.h"
#include "glPushContext.h"

#include <QOpenGLShaderProgram>

namespace Heightmap {
namespace Render {

RenderRegion::RenderRegion(glProjection gl_projection)
    :
      gl_projection_(gl_projection)
{
}


RenderRegion::~RenderRegion()
{
    delete program_;
}


void RenderRegion::
        render(Region r, bool drawcross)
{
    // if (!renderBlock(...) && (0 == "render red warning cross" || render_settings->y_scale < yscalelimit))
    //float y = _frustum_clip.projectionPlane[1]*.05;
    float y = 0.05f;

    if (!program_) {
        program_ = new QOpenGLShaderProgram();
        program_->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                           "attribute highp vec4 vertices;"
                                           "uniform highp mat4 modelviewproj;"
                                           "void main() {"
                                           "    gl_Position = modelviewproj*vertices;"
                                           "}");
        program_->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                           "uniform lowp vec4 color;"
                                           "void main() {"
                                           "    gl_FragColor = color;"
                                           "}");

        program_->bindAttributeLocation("vertices", 0);
        program_->link();
    }

    program_->bind();

    program_->enableAttributeArray(0);

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

    matrixd modelview = gl_projection_.modelview;
    modelview *= matrixd::translate (r.a.time, 0, r.a.scale);
    modelview *= matrixd::scale (r.time(), 1, r.scale());

    glEnable(GL_BLEND);
    glBindTexture(GL_TEXTURE_2D, 0);
    glLineWidth(2);

    program_->setUniformValue("color", 0.8, 0.2, 0.2, 0.5);
    program_->setUniformValue("modelviewproj",
                              QMatrix4x4(GLmatrixf(gl_projection_.projection*modelview).transpose ().v ()));
    glDrawArrays(GL_LINE_STRIP, 0, n_values);

    program_->setUniformValue("color", 0.2, 0.8, 0.8, 0.5);
    program_->setUniformValue("modelviewproj",
                              QMatrix4x4(GLmatrixf(gl_projection_.projection*matrixd::translate (0,y,0)*modelview).transpose ().v ()));
    glDrawArrays(GL_LINE_STRIP, 0, n_values);

    program_->disableAttributeArray(0);
    program_->release();
}

} // namespace Render
} // namespace Heightmap
