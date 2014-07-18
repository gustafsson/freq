#include "squirclerenderer.h"
#include "tools/support/heightmapprocessingpublisher.h"
#include "tools/support/renderviewupdateadapter.h"
#include "signal/processing/chain.h"

SquircleRenderer::SquircleRenderer()
    :
      render_view(&render_model),
      m_t(0), m_program(0)
{
    render_model.resetCameraSettings ();

    Tools::Support::RenderViewUpdateAdapter* rvup;
    Tools::Support::RenderOperationDesc::RenderTarget::ptr rvu(
                rvup = new Tools::Support::RenderViewUpdateAdapter);

    connect(rvup, SIGNAL(redraw()), &render_view, SLOT(redraw()));

    render_model.init(Signal::Processing::Chain::createDefaultChain (), rvu);

    // 'this' is parent
    auto hpp = new Tools::Support::HeightmapProcessingPublisher(
                render_model.target_marker ()->target_needs (),
                render_model.tfr_mapping (),
                &render_model.camera.q[0],
                this);
    connect(rvup, SIGNAL(setLastUpdatedInterval(Signal::Interval)), hpp, SLOT(setLastUpdatedInterval(Signal::Interval)));
    connect(&render_view, SIGNAL(painting()), hpp, SLOT(update()));
}

SquircleRenderer::~SquircleRenderer()
{
    delete m_program;
}

/*
void SquircleRenderer::paint()
{
    if (!m_program) {
        m_program = new QOpenGLShaderProgram();
        m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                           "attribute highp vec4 vertices;"
                                           "varying highp vec2 coords;"
                                           "void main() {"
                                           "    gl_Position = vertices;"
                                           "    coords = vertices.xy;"
                                           "}");
        m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                           "uniform lowp float t;"
                                           "varying highp vec2 coords;"
                                           "void main() {"
                                           "    lowp float i = 1. - (pow(abs(coords.x), 4.) + pow(abs(coords.y), 4.));"
                                           "    i = smoothstep(t - 0.8, t + 0.8, i);"
                                           "    i = floor(i * 20.) / 20.;"
                                           "    gl_FragColor = vec4(coords * .5 + .5, i, i);"
                                           "}");

        m_program->bindAttributeLocation("vertices", 0);
        m_program->link();

    }

    m_program->bind();

    m_program->enableAttributeArray(0);

    float values[] = {
     -1, -1,
     1, -1,
     -1, 1,
     1, 1
    };
    m_program->setAttributeArray(0, GL_FLOAT, values, 2);
    m_program->setUniformValue("t", (float) m_t);

    glViewport(0, 0, m_viewportSize.width(), m_viewportSize.height());

    glDisable(GL_DEPTH_TEST);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    m_program->disableAttributeArray(0);
    m_program->release();
 }
*/
void SquircleRenderer::paint()
{
    render_view.resizeGL (QRect(QPoint(0,0),m_viewportSize), m_viewportSize.height ());
    render_view.initializeGL ();
    render_view.setStates ();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    render_view.paintGL ();
}
