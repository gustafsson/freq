#include "squirclerenderer.h"
#include "signal/processing/chain.h"
#include "signal/processing/workers.h"
#include "heightmap/collection.h"
#include "log.h"
#include "tasktimer.h"
#include "GlException.h"
#include "demangle.h"
#include "heightmap/render/shaderresource.h"
#include "glgroupmarker.h"

#include <boost/exception/exception.hpp>
#include <QTimer>

//#define LOG_FRAME
#define LOG_FRAME if(0)

SquircleRenderer::SquircleRenderer(Tools::RenderModel* render_model)
    :
      render_view(render_model),
      m_t(0)
{
    connect(&render_view, SIGNAL(redrawSignal()), this, SIGNAL(redrawSignal()));
}


SquircleRenderer::~SquircleRenderer()
{
}


void SquircleRenderer::
        setViewport(const QRectF &rect, const QSize& window, double ratio)
{
    render_view.model->render_settings.dpifactor = ratio;
    m_viewport = QRectF(rect.topLeft ()*ratio, rect.bottomRight ()*ratio).toRect ();
    m_window = window*ratio;
    if (m_viewport.height ()>0 && m_viewport.width ()>0)
        render_view.resizeGL (m_viewport, m_window);
}


void SquircleRenderer::paint3()
{
    if (!vertexbuffer)
    {
        // An array of 3 vectors which represents 3 vertices
        static const GLfloat g_vertex_buffer_data[] = {
           -1.0f, -1.0f, 0.0f,
           1.0f, -1.0f, 0.0f,
           0.0f,  1.0f, 0.0f,
        };

        // This will identify our vertex buffer
        // vertexbuffer;

        // Generate 1 buffer, put the resulting identifier in vertexbuffer
        GlException_SAFE_CALL( glGenBuffers(1, &vertexbuffer) );

        // The following commands will talk about our 'vertexbuffer' buffer
        GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer) );

        // Give our vertices to OpenGL.
        GlException_SAFE_CALL( glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW) );

        m_program = Heightmap::ShaderResource::loadGLSLProgramSource (
                                           R"vertexshader(
                                               attribute vec3 vertexPosition_modelspace;
                                               void main() {
                                                   gl_Position.xyz = vertexPosition_modelspace;
                                                   gl_Position.w = 1.0;
                                               }
                                           )vertexshader",
                                           R"fragmentshader(
                                               void main(){
                                                   gl_FragColor = vec4(1,0,0,1);
                                               }
                                            )fragmentshader");
    }

    if (!m_program->isLinked ())
        return;

    static int frame_count=0;
    Log("frame_count: %d") % ++frame_count;

    GlException_SAFE_CALL( glClearColor (0,1,0,1) );
    GlException_SAFE_CALL( glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) );

    GlException_SAFE_CALL( glEnableVertexAttribArray(0) );
    GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer) );
    GlException_SAFE_CALL( glVertexAttribPointer(
       0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
       3,                  // size
       GL_FLOAT,           // type
       GL_FALSE,           // normalized?
       0,                  // stride
       (void*)0            // array buffer offset
    ));

    // Draw the triangle !
    GlException_SAFE_CALL( glUseProgram(m_program->programId ()) );
    GlException_SAFE_CALL( glDrawArrays(GL_TRIANGLES, 0, 3) ); // Starting from vertex 0; 3 vertices total -> 1 triangle

    GlException_SAFE_CALL( glDisableVertexAttribArray(0) );
}

void SquircleRenderer::paint2()
{
//    Log("version: %s") % glGetString(GL_VERSION);
//    Log("Current context: %d.%d")
//            % QOpenGLContext::currentContext ()->format ().majorVersion ()
//            % QOpenGLContext::currentContext ()->format ().minorVersion ();

    if (!m_program) {
        static const GLfloat values[] = {
         -1, -1,
         1, -1,
         -1, 1,
         1, 1
        };

        // Generate 1 buffer, put the resulting identifier in vertexbuffer
        GlException_SAFE_CALL( glGenBuffers(1, &vertexbuffer) );

        // The following commands will talk about our 'vertexbuffer' buffer
        GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer) );

        // Give our vertices to OpenGL.
        GlException_SAFE_CALL( glBufferData(GL_ARRAY_BUFFER, sizeof(values), values, GL_STATIC_DRAW) );

        m_program = Heightmap::ShaderResource::loadGLSLProgramSource (
                                           R"vertexshader(
                                               attribute highp vec4 vertices;
                                               varying highp vec2 coords;
                                               void main() {
                                                   gl_Position = vertices;
                                                   coords = vertices.xy;
                                               }
                                           )vertexshader",
                                           R"fragmentshader(
                                               uniform lowp float t;
                                               varying highp vec2 coords;
                                               void main() {
                                                   lowp float i = 1. - (pow(abs(coords.x), 4.) + pow(abs(coords.y), 4.));
                                                   i = smoothstep(t - 0.8, t + 0.8, i);
                                                   i = floor(i * 20.) / 20.;
                                                   gl_FragColor = vec4(coords * .5 + .5, i, i);
                                               }
                                           )fragmentshader");
    }
    if (!m_program->isLinked())
        return;

    GlException_SAFE_CALL( m_program->bind() );

    GlException_SAFE_CALL( m_program->enableAttributeArray(0) );

    GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer) );
    GlException_SAFE_CALL( m_program->setAttributeBuffer(0, GL_FLOAT, 0, 2) );
    GlException_SAFE_CALL( m_program->setUniformValue("t", (float) m_t) );

    GlException_SAFE_CALL( glViewport(m_viewport.x(), m_window.height () - m_viewport.y() - m_viewport.height(), m_viewport.width(), m_viewport.height()) );

    GlException_SAFE_CALL( glDisable(GL_DEPTH_TEST) );

    GlException_SAFE_CALL( glEnable(GL_BLEND) );
    GlException_SAFE_CALL( glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) );

    GlException_SAFE_CALL( glDrawArrays(GL_TRIANGLE_STRIP, 0, 4) );

    GlException_SAFE_CALL( m_program->disableAttributeArray(0) );
    GlException_SAFE_CALL( m_program->release() );
}


void SquircleRenderer::paint()
{
    static bool failed = false;
    if (failed)
        return;

    GlGroupMarker gpm("SquircleRenderer");

    try {

        GlException_CHECK_ERROR();

        if (m_viewport.height ()==0 || m_viewport.width ()==0)
        {
            for ( auto c : renderView ()->model->tfr_mapping ()->collections() )
                c->frame_begin(); // increment frame_number and keep garbage collection running
            return;
        }

        LOG_FRAME TaskTimer tt(boost::format("painting %s %gx%g. Last frame %s ms")
                % objectName ().toStdString ()
                % m_viewport.width ()
                % m_viewport.height ()
                % (1e3*prevFrame.elapsedAndRestart ()));

        // TODO: Use WorkerCrashLogger from Sonic AWE instead
        try {
            auto c = this->render_view.model->chain ();
            if (c) c->workers()->rethrow_any_worker_exception();
        } catch(const boost::exception&x) {
            Log("%s") % boost::diagnostic_information(x);
        }

        GlException_SAFE_CALL( render_view.initializeGL () );

        GlException_SAFE_CALL( render_view.setStates () );

        GlException_SAFE_CALL( render_view.paintGL () );

    } catch (const ExceptionAssert& x) {
        char const * const * f = boost::get_error_info<boost::throw_file>(x);
        int const * l = boost::get_error_info<boost::throw_line>(x);
        char const * const * c = boost::get_error_info<ExceptionAssert::ExceptionAssert_condition>(x);
        std::string const * m = boost::get_error_info<ExceptionAssert::ExceptionAssert_message>(x);

        fflush(stdout);
        fprintf(stderr, "%s",
                (boost::format("%s:%d: %s. %s\n"
                                  "%s\n"
                                  " FAILED in %s\n\n")
                    % (f?*f:0) % (l?*l:-1) % (c?*c:0) % (m?*m:0) % boost::diagnostic_information(x) % __FUNCTION__ ).str().c_str());
        fflush(stderr);
        failed = true;
    } catch (const std::exception& x) {
        fflush(stdout);
        fprintf(stderr, "%s",
                (boost::format("%s\n"
                                  "%s\n"
                                  " FAILED in %s\n\n")
                    % vartype(x) % boost::diagnostic_information(x) % __FUNCTION__ ).str().c_str());
        fflush(stderr);
        failed = true;
    } catch (...) {
        fflush(stdout);
        fprintf(stderr, "%s",
                (boost::format("Not an std::exception\n"
                                  "%s\n"
                                  " FAILED in %s\n\n")
                    % boost::current_exception_diagnostic_information () % __FUNCTION__ ).str().c_str());
        fflush(stderr);
        failed = true;
    }
}
