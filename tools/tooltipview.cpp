#include "tooltipview.h"
#include "commentview.h"
#include "tooltipcontroller.h"

//#include <sawe/project.h>

#include <glPushContext.h>

namespace Tools {


TooltipView::
        TooltipView(TooltipModel* model,
                    TooltipController* controller,
                    RenderView* render_view)
                        :
                        enabled(true),
                        visible(true),
                        initialized(false),
                        model_(model),
                        controller_(controller),
                        render_view_(render_view)
{
    connect(render_view_, SIGNAL(painting()), SLOT(draw()));
}


TooltipView::
        ~TooltipView()
{
    model_->removeFromRepo();
}


void TooltipView::
        drawMarkers()
{
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
        setHidden(bool value)
{
    visible = !value;
}


void TooltipView::
        setFocus()
{
    controller_->setCurrentView( this );
}


void TooltipView::
        seppuku()
{
    render_view_->userinput_update();
    delete this;
}


void TooltipView::
        initialize()
{
    initialized = true;

    connect(model_->comment.data(), SIGNAL(thumbnailChanged(bool)), SLOT(setHidden(bool)));
    connect(model_->comment.data(), SIGNAL(gotFocus()), SLOT(setFocus()));
    connect(model_->comment.data(), SIGNAL(destroyed()), SLOT(seppuku()));
}


void TooltipView::
        draw()
{
    if (!model_->comment)
        return;

    if (!initialized)
        initialize();

    //if (visible)
    {
        drawMarkers();

        if (model_->automarking == TooltipModel::AutoMarkerWorking)
        {
            TaskInfo ti("TooltipView doesn't have all data yet");

            model_->showToolTip( model_->pos );

            TaskInfo("%s", model_->comment->html().c_str());
            if (model_->automarking != TooltipModel::AutoMarkerWorking)
                TaskInfo("TooltipView just finished");

            render_view_->userinput_update( false );

            emit tooltipChanged();
        }
    }
}

} // namespace Tools
