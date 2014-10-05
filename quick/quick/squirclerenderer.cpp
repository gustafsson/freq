#include "squirclerenderer.h"
#include "signal/processing/chain.h"
#include "signal/processing/workers.h"
#include "log.h"
#include <boost/exception/exception.hpp>
#include <QTimer>

SquircleRenderer::SquircleRenderer(Tools::RenderModel* render_model)
    :
      render_view(render_model),
      m_t(0), m_program(0)
{
    connect(&render_view, SIGNAL(redrawSignal()), this, SIGNAL(redrawSignal()));
}


SquircleRenderer::~SquircleRenderer()
{
    delete m_program;
}


void SquircleRenderer::
        setViewport(const QRectF &rect, double window_height, double ratio)
{
    m_viewport = QRectF(rect.topLeft ()*ratio, rect.bottomRight ()*ratio).toRect ();
    m_window_height = window_height*ratio;
    render_view.resizeGL (m_viewport, m_window_height);
    render_view.initializeGL ();
}


void SquircleRenderer::paint2()
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

    glViewport(m_viewport.x(), m_window_height - m_viewport.y() - m_viewport.height(), m_viewport.width(), m_viewport.height());

    glDisable(GL_DEPTH_TEST);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    m_program->disableAttributeArray(0);
    m_program->release();
}


void SquircleRenderer::paint()
{
    // Use WorkerCrashLogger from Sonic AWE instead
    try {
        auto c = this->render_view.model->chain ();
        if (c) c->workers()->rethrow_any_worker_exception();
    } catch(const boost::exception&x) {
        Log("%s") % boost::diagnostic_information(x);
    }

    render_view.setStates ();
    const auto c = *render_view.model->camera.read ();
    render_view.model->recompute_extent ();
    double sL = render_view.model->tfr_mapping ()->length();

    if (c.q[0] > sL-0.01) {
        render_view.model->camera->q[0] = sL;
        emit repositionSignal();
    }

    // neat if no other viewport tries to modify this camera position by the same
    // algorithm
    //    double s = 4*m_viewport.width ()
    //            /(float)m_viewport.height ()
    //            /c.xscale
    //            *sin(DEG_TO_RAD(c.r[0]));
    //    double L = sL - s;
    //    static double lq0 = c.q[0];
    //    if (c.q[0] > L-0.1*s || c.q[0] == lq0)
    //        render_view.model->camera->q[0] = lq0 = L;

    render_view.paintGL ();
}

