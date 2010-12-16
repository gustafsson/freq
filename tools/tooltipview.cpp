#include "tooltipview.h"

#include <glPushContext.h>

namespace Tools {


TooltipView::
        TooltipView(TooltipModel* model,
                    RenderView* render_view)
                        :
                        enabled(true),
                        visible(true),
                        model_(model),
                        render_view_(render_view)
{
}


TooltipView::
        ~TooltipView()
{
}


void TooltipView::
        drawMarkers()
{
    if (!model_->comment)
        return;

    Heightmap::Position p = model_->pos;

    const Tfr::FreqAxis& display_scale = render_view_->model->display_scale();
    double frequency = display_scale.getFrequency( p.scale );
    double fundamental_frequency = frequency / model_->markers;
    for (unsigned i=1; i<=2*model_->markers || i <= 10; ++i)
    {
        float harmonic = fundamental_frequency * i;
        p.scale = display_scale.getFrequencyScalar( harmonic );

        if (p.scale>=1)
            break;

        if (p.scale>=0)
            drawMarker( p );
    }
}


void TooltipView::
        drawMarker( Heightmap::Position p )
{
    glPushAttribContext ac;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);
    glColor4f( 0.7, 0.2, 0.2, enabled ? .5 : 0.2);

    float sz = -0.01/render_view_->model->xscale*render_view_->model->_pz;
    float x1 = p.time - sz;
    float x2 = p.time + sz;
    float y = 1.5;

    glBegin(GL_TRIANGLE_STRIP);
        glVertex3f( x1, 0, p.scale );
        glVertex3f( x1, y, p.scale );
        glVertex3f( x2, 0, p.scale );
        glVertex3f( x2, y, p.scale );
    glEnd();

    glLineWidth(1.6f);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_LINE_STRIP);
        glVertex3f( x1, 0, p.scale );
        glVertex3f( x1, y, p.scale );
        glVertex3f( x2, y, p.scale );
        glVertex3f( x2, 0, p.scale );
    glEnd();
    glLineWidth(0.5f);
    glDepthMask(true);
}


void TooltipView::
        draw()
{
    if (visible)
        drawMarkers();
}

} // namespace Tools
