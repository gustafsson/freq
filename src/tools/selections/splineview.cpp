#include "splineview.h"
#include "splinemodel.h"

#include "tools/support/toolglbrush.h"

#include "tasktimer.h"

namespace Tools { namespace Selections
{


SplineView::SplineView(SplineModel* model)
    :
    visible(true),
    enabled(false),
    model_(model)
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
    if (visible)
        drawSelectionSpline();
}


void SplineView::
        drawSelectionSpline()
{
#ifdef LEGACY_OPENGL
    float y = 1;

    Support::ToolGlBrush tgb(enabled);
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
        glBegin( model_->drawing ? GL_LINE_STRIP : GL_LINE_LOOP );
            for (unsigned i=0; i<model_->v.size(); ++i )
            {
                glVertex3f( model_->v[i].time, y, model_->v[i].scale );
            }
        glEnd();
        glLineWidth(0.5f);
    }
#endif // LEGACY_OPENGL
}


}} // namespace Tools::Selections
