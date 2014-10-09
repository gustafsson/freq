#include "selectionrenderer.h"
#include "log.h"

SelectionRenderer::SelectionRenderer(SquircleRenderer* parent) :
    QObject(parent)
{
    connect(parent->renderView (), SIGNAL(painting()), SLOT(painting()), Qt::DirectConnection);
    model = parent->renderView ()->model;
}


SelectionRenderer::~SelectionRenderer()
{
    delete m_program;
}


void SelectionRenderer::
        setSelection(double t1, double f1, double t2, double f2)
{
    this->t1 = t1;
    this->f1 = f1;
    this->t2 = t2;
    this->f2 = f2;
}


void SelectionRenderer::
        painting()
{
    if (t1==t2 || f1==f2)
        return;

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
            Log("selectionrenderer: invalid shader\n%s")
                    % m_program->log ().toStdString ();
    }

    if (!m_program->isLinked ())
        return;

    m_program->bind();

    m_program->enableAttributeArray(0);

    float h1 = -100;
    float h2 = 100;
    Heightmap::FreqAxis f = model->tfr_mapping ().read ()->display_scale();
    float s1 = f.getFrequencyScalar (f1);
    float s2 = f.getFrequencyScalar (f2);

    float values[] = {
        0, 0, 0, // counter clock-wise front sides, but culling isn't enabled
        0, 1, 0,
        0, 0, 1,
        0, 1, 1,
        1, 0, 1,
        1, 1, 1,
        1, 0, 0,
        1, 1, 0,
        0, 0, 0,
        0, 1, 0,
        0, 1, 0, // degenerate, and start of top
        1, 1, 0,
        0, 1, 1,
        1, 1, 1,
        1, 1, 1, // degenerate, stop top
        0, 0, 0, // degenerate, to bottom
        0, 0, 0, // degenerate, and start of bottom
        0, 0, 1,
        1, 0, 0,
        1, 0, 1,
    };

    const int N = sizeof(values)/sizeof(values[0])/3;

    m_program->setAttributeArray(0, GL_FLOAT, values, 3);
    glProjection p = *model->gl_projection.read ();
    matrixd modelview = p.modelview;
    modelview *= matrixd::translate (t1,h1,s1);
    modelview *= matrixd::scale (t2-t1,h2-h1,s2-s1);
    QMatrix4x4 modelviewprojection {GLmatrixf(p.projection*modelview).transpose ().v ()};
    m_program->setUniformValue("ModelViewProjectionMatrix", modelviewprojection);
    m_program->setUniformValue("rgba", QVector4D(0.0,0.0,0.0,0.5));

    // don't write to the depth buffer
    glDepthMask(GL_FALSE);
    glEnable (GL_STENCIL_TEST); // must enable testing for glStencilOp(INVERT) to take effect

    // write to the stencil buffer but not to the color buffer
    glStencilMask(0xFF);
    glColorMask (GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);

    // set the stencil buffer to 1 where objects in the current scene are inside the selection box
    glClear (GL_STENCIL_BUFFER_BIT);
    glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
    glStencilFunc(GL_ALWAYS, 1, 1);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, N); // <- draw call

    glDisable (GL_DEPTH_TEST);
    // this flips the stencil once if the camera is inside the selection box
    // and leaves it unaffected by flipping it twice if the camera is outside of the selection box
    glDrawArrays(GL_TRIANGLE_STRIP, 0, N); // <- draw call
    glEnable (GL_DEPTH_TEST);

    // write to the color buffer but not to the stencil buffer
    glColorMask (GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);

    // draw where exactly one fragment of the selection box was visible
    // flip the stencil bit when the fragment has been drawn
    glStencilFunc(GL_EQUAL, 1, 1);
    glDisable (GL_DEPTH_TEST); // disabled depth test needed if the camera is inside the selection box
    glDrawArrays (GL_TRIANGLE_STRIP, 0, N); // <- draw call
    glEnable (GL_DEPTH_TEST);

    // write to the depth buffer again and stop fiddling with the stencil buffer
    glDepthMask(GL_TRUE);
    glDisable (GL_STENCIL_TEST);

    m_program->disableAttributeArray(0);
    m_program->release();
}
