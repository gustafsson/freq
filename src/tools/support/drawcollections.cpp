#include "drawcollections.h"
#include "GlException.h"
#include "channelcolors.h"
#include "computationkernel.h"
#include "glPushContext.h"

//#define TIME_PAINTGL_DRAW
#define TIME_PAINTGL_DRAW if(0)

//#define TIME_PAINTGL_DETAILS
#define TIME_PAINTGL_DETAILS if(0)

namespace Tools {
namespace Support {

DrawCollections::DrawCollections(RenderModel* model)
    :
      model(model)
{
}


void DrawCollections::
        drawCollections(GlFrameBuffer* fbo, float yscale)
{
    TIME_PAINTGL_DRAW TaskTimer tt2("Drawing...");
    GlException_CHECK_ERROR();

    unsigned N = model->collections().size();
    if (N != channel_colors.size ())
        channel_colors = Support::ChannelColors::compute(N);
    TIME_PAINTGL_DETAILS ComputationCheckError();

    // Draw the first channel without a frame buffer
    model->renderer->render_settings.camera = GLvector(model->_qx, model->_qy, model->_qz);
    model->renderer->render_settings.cameraRotation = GLvector(model->_rx, model->_ry, model->_rz);

    // When rendering to fbo, draw to the entire fbo, then update the current
    // viewport.
    GLint current_viewport[4];
    glGetIntegerv(GL_VIEWPORT, current_viewport);
    GLint viewportWidth = current_viewport[2],
          viewportHeight = current_viewport[3];


    TIME_PAINTGL_DETAILS TaskTimer tt("Viewport (%u, %u, %u, %u)",
        current_viewport[0], current_viewport[1],
        current_viewport[2], current_viewport[3]);

    unsigned i=0;

    const Heightmap::TfrMapping::Collections collections = model->collections ();

    // draw the first without fbo
    for (; i < N; ++i)
    {
        if (!collections[i].read ()->isVisible())
            continue;

        drawCollection(i, yscale);
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
            if (viewportWidth > fbo->getWidth ()
                || viewportHeight > fbo->getHeight()
                || viewportWidth*2 < fbo->getWidth()
                || viewportHeight*2 < fbo->getHeight())
            {
                fbo->recreate(viewportWidth*1.5, viewportHeight*1.5);
            }

            hasValidatedFboSize = true;
        }

        GlException_CHECK_ERROR();

        {
            GlFrameBuffer::ScopeBinding fboBinding = fbo->getScopeBinding();
            glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
            glViewport(0, 0, viewportWidth, viewportHeight);

            drawCollection(i, yscale);
        }

        glViewport(current_viewport[0], current_viewport[1],
                   current_viewport[2], current_viewport[3]);

        glPushMatrixContext mpc( GL_PROJECTION );
        glLoadIdentity();
        glOrtho(0,1,0,1,-10,10);
        glPushMatrixContext mc( GL_MODELVIEW );
        glLoadIdentity();

        glBlendFunc( GL_DST_COLOR, GL_ZERO );

        glDisable(GL_DEPTH_TEST);

        glColor4f(1,1,1,1);
        GlTexture t(fbo->getGlTexture(), fbo->getWidth (), fbo->getHeight ());
        GlTexture::ScopeBinding texObjBinding = t.getScopeBinding();

        glBegin(GL_TRIANGLE_STRIP);
            float tx = viewportWidth/(float)fbo->getWidth();
            float ty = viewportHeight/(float)fbo->getHeight();
            glTexCoord2f(0,0); glVertex2f(0,0);
            glTexCoord2f(tx,0); glVertex2f(1,0);
            glTexCoord2f(0,ty); glVertex2f(0,1);
            glTexCoord2f(tx,ty); glVertex2f(1,1);
        glEnd();

        glEnable(GL_DEPTH_TEST);

        GlException_CHECK_ERROR();
    }

    TIME_PAINTGL_DETAILS ComputationCheckError();
    TIME_PAINTGL_DETAILS GlException_CHECK_ERROR();

    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    TIME_PAINTGL_DRAW
    {
        unsigned collections_n = 0;
        for (i=0; i < N; ++i)
            collections_n += collections[i].read ()->isVisible();

        TaskInfo("Drew %u channels*%u block%s*%u triangles (%u triangles in total) in viewport(%d, %d).",
        collections_n,
        model->renderer->render_settings.drawn_blocks,
        model->renderer->render_settings.drawn_blocks==1?"":"s",
        model->renderer->trianglesPerBlock(),
        collections_n*model->renderer->render_settings.drawn_blocks*model->renderer->trianglesPerBlock(),
        current_viewport[2], current_viewport[3]);
    }
}


void DrawCollections::
        drawCollection(int i, float yscale )
{
    model->renderer->collection = model->collections()[i];
    model->renderer->render_settings.fixed_color = channel_colors[i];
    glDisable(GL_BLEND);
    if (0 != model->_rx)
        glEnable( GL_CULL_FACE ); // enabled only while drawing collections
    else
        glEnable( GL_DEPTH_TEST );
    float L = model->tfr_mapping().read()->length();
    model->renderer->draw( yscale, L ); // 0.6 ms
    glDisable( GL_CULL_FACE );
    glEnable(GL_BLEND);
}


} // namespace Support
} // namespace Tools
