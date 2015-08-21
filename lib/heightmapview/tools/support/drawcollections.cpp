#include "drawcollections.h"

#include "glstate.h"
#include "GlException.h"
#include "computationkernel.h"
#include "glPushContext.h"
#include "tasktimer.h"

#include "channelcolors.h"
#include "tools/rendermodel.h"
#include "heightmap/render/renderer.h"

#include <QOpenGLShaderProgram>

//#define TIME_PAINTGL_DRAW
#define TIME_PAINTGL_DRAW if(0)

//#define DRAW_INFO
#define DRAW_INFO if(0)

//#define TIME_PAINTGL_DETAILS
#define TIME_PAINTGL_DETAILS if(0)

namespace Tools {
namespace Support {

DrawCollections::DrawCollections(RenderModel* model)
    :
      model(model),
      render_block(&model->render_settings)
{
}


DrawCollections::~DrawCollections()
{
    delete m_program;
}


void DrawCollections::
        drawCollections(const glProjection& gl_projection, GlFrameBuffer* fbo, float yscale)
{
    TIME_PAINTGL_DRAW TaskTimer tt2("Drawing...");
    GlException_CHECK_ERROR();

    render_block.init();
    if (!render_block.isInitialized())
        return;

    unsigned N = model->collections().size();
    bool fixed_color = model->render_settings.color_mode == Heightmap::Render::RenderSettings::ColorMode_FixedColor;
    if (N - !fixed_color != channel_colors.size ())
        channel_colors = Support::ChannelColors::compute(N - !fixed_color);
    TIME_PAINTGL_DETAILS ComputationCheckError();

    // When rendering to fbo, draw to the entire fbo, then update the current
    // viewport.
    const auto vp = gl_projection.viewport;

    TIME_PAINTGL_DETAILS TaskTimer tt("Viewport (%u, %u, %u, %u)",
        vp[0], vp[1],
        vp[2], vp[3]);

    unsigned i=0;

    const Heightmap::TfrMapping::Collections collections = model->collections ();

    // Draw the first channel without a frame buffer
    for (; i < N; ++i)
    {
        if (!collections[i].read ()->isVisible())
            continue;

        drawCollection(gl_projection, i, yscale);
        ++i;
        break;
    }


    bool hasValidatedFboSize = false;

    for (; i<N; ++i)
    {
        if (!collections[i].read ()->isVisible())
            continue;

        if (!hasValidatedFboSize)
        {
            // drawCollections is called for several different viewports each frame.
            // GlFrameBuffer will query the current viewport to determine the size
            // of the fbo for this iteration.
            if (vp[2] > fbo->getWidth ()
                || vp[3] > fbo->getHeight()
                || vp[2]*2 < fbo->getWidth()
                || vp[3]*2 < fbo->getHeight())
            {
                TaskInfo("new fbo");
                fbo->recreate(vp[2]*1.5, vp[3]*1.5);
            }

            hasValidatedFboSize = true;
        }

        GlException_CHECK_ERROR();

        {
            GlFrameBuffer::ScopeBinding fboBinding = fbo->getScopeBinding();
            glViewport(0, 0, vp[2], vp[3]);
            glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
            auto gl_projection2 = gl_projection;
            gl_projection2.viewport = tvector<4,int>(0, 0, vp[2], vp[3]);

            drawCollection(gl_projection, i, yscale);
        }

        glViewport(vp[0], vp[1],
                   vp[2], vp[3]);

        glBlendFunc( GL_DST_COLOR, GL_ZERO );

        GlState::glDisable (GL_DEPTH_TEST);
        GlTexture t(fbo->getGlTexture(), fbo->getWidth (), fbo->getHeight ());
        t.bindTexture();

        if (!m_program) {
            m_program = new QOpenGLShaderProgram();
            m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                               "attribute highp vec4 vertices;"
                                               "attribute highp vec2 itex;"
                                               "uniform mat4 modelviewprojection;"
                                               "varying highp vec2 ftex;"
                                               "void main() {"
                                               "    gl_Position = modelviewprojection * vertices;"
                                               "    ftex = itex;"
                                               "}");
            m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                               "uniform lowp float t;"
                                               "uniform sampler2D tex;"
                                               "varying highp vec2 ftex;"
                                               "void main() {"
                                               "    gl_FragColor = texture2D(tex, ftex);"
                                               "}");

            m_program->link();
            m_program->bind();
            m_program->setUniformValue ("tex", 0);
            QMatrix4x4 M;
            M.ortho (0,1,0,1,-10,10);
            m_program->setUniformValue ("modelviewprojection", M);
            m_program->release();
        }

        m_program->bind();
        GlState::glEnableVertexAttribArray (0);
        GlState::glEnableVertexAttribArray (1);

        float tx = vp[2]/(float)fbo->getWidth();
        float ty = vp[3]/(float)fbo->getHeight();
        float values[] = {
             0, 0, 0,  0,
             1, 0, tx, 0,
             0, 1, 0,  ty,
             1, 1, tx, ty
        };
        m_program->setAttributeArray("vertices", GL_FLOAT, values, 2, 4*sizeof(float));
        m_program->setAttributeArray("itex", GL_FLOAT, values + 2, 2, 4*sizeof(float));
        GlState::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        GlState::glDisableVertexAttribArray (1);
        GlState::glDisableVertexAttribArray (0);
        m_program->release();

        GlState::glEnable (GL_DEPTH_TEST);

        GlException_CHECK_ERROR();
    }

    TIME_PAINTGL_DETAILS ComputationCheckError();
    TIME_PAINTGL_DETAILS GlException_CHECK_ERROR();

    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    DRAW_INFO
    {
        unsigned collections_n = 0;
        for (i=0; i < N; ++i)
            collections_n += collections[i].read ()->isVisible();

        TaskInfo("Drew %u channels*%u block%s*%u triangles (%u triangles in total) in viewport(%d, %d).",
                collections_n,
                model->render_settings.drawn_blocks,
                model->render_settings.drawn_blocks==1?"":"s",
                render_block.trianglesPerBlock(),
                collections_n*model->render_settings.drawn_blocks*render_block.trianglesPerBlock(),
                vp[2], vp[3]);
    }
}


void DrawCollections::
        drawCollection(const glProjection& gl_projection, int i, float yscale )
{
    bool fixed_color = model->render_settings.color_mode == Heightmap::Render::RenderSettings::ColorMode_FixedColor;
    if (fixed_color || 0<i)
        model->render_settings.fixed_color = channel_colors[fixed_color ? i : std::max(0,i-1)];
    GlState::glDisable (GL_BLEND);
    if (0 != model->camera->r[0])
        GlState::glEnable ( GL_CULL_FACE ); // enabled only while drawing collections
    else
        GlState::glEnable ( GL_DEPTH_TEST );
    float L = model->tfr_mapping().read()->length();

    Heightmap::Render::Renderer renderer(model->collections()[i],
                                         model->render_settings,
                                         gl_projection,
                                         &render_block);
    renderer.draw( yscale, L ); // 0.6 ms

    GlState::glDisable ( GL_CULL_FACE );
    GlState::glEnable (GL_BLEND);
}


} // namespace Support
} // namespace Tools
