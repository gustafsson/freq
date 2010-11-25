#include "splineview.h"
#include "splinemodel.h"

#ifdef _MSC_VER // gl.h expects windows.h to be included on windows
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <GL/gl.h>
#include <TaskTimer.h>
#include <glPushContext.h>

namespace Tools { namespace Selections
{


SplineView::SplineView(SplineModel* model, Signal::Worker* worker)
    :
    enabled(false),
    model_(model),
    worker_(worker)
{
}


SplineView::
        ~SplineView()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void SplineView::
        draw()
{
    drawSelectionSpline();
}


void SplineView::
        drawSelectionSpline()
{
    float y = 1;

    glPushAttribContext ac;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);
    glColor4f( 0, 0, 0, enabled ? .5 : 0.2);

    if (!model_->v.empty())
    {
        glBegin(GL_TRIANGLE_STRIP);
        for (unsigned i=0; i<model_->v.size(); ++i )
        {
            glVertex3f( model_->v[i].time, y, model_->v[i].scale );
            glVertex3f( model_->v[i].time, 0, model_->v[i].scale );
        }
        if (!model_->drawing)
        {
            glVertex3f( model_->v[0].time, y, model_->v[0].scale );
            glVertex3f( model_->v[0].time, 0, model_->v[0].scale );
        }
        glEnd();

        glLineWidth( 1.6f );
        glPolygonOffset( 1.f, 1.f );
        glBegin( model_->drawing ? GL_LINE_STRIP : GL_LINE_LOOP );
            for (unsigned i=0; i<model_->v.size(); ++i )
            {
                glVertex3f( model_->v[i].time, y, model_->v[i].scale );
            }
        glEnd();
        glLineWidth(0.5f);
    }
    glDepthMask(true);
}


}} // namespace Tools::Selections
