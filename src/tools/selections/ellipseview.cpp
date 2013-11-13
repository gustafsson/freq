#include "ellipseview.h"
#include "ellipsemodel.h"

#include "tools/support/toolglbrush.h"

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "TaskTimer.h"

namespace Tools { namespace Selections
{


EllipseView::EllipseView(EllipseModel* model)
    :
    visible(false),
    enabled(false),
    model_(model)
{
}


EllipseView::
        ~EllipseView()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void EllipseView::
        draw()
{
    if (visible)
        drawSelectionCircle();
}

void EllipseView::
        drawSelectionCircle()
{
    float
        x = model_->centre.time,
        z = model_->centre.scale,
        _rx = fabs(model_->centrePlusRadius.time - model_->centre.time),
        _rz = fabs(model_->centrePlusRadius.scale - model_->centre.scale);
    float y = 1;

    Support::ToolGlBrush tgb(enabled);

    glBegin(GL_TRIANGLE_STRIP);
    for (unsigned k=0; k<=360; k++) {
        float s = z + _rz*sin(k*M_PI/180);
        float c = x + _rx*cos(k*M_PI/180);
        glVertex3f( c, 0, s );
        glVertex3f( c, y, s );
    }
    glEnd();

    glLineWidth(1.6f);
    glBegin(GL_LINE_LOOP);
    for (unsigned k=0; k<360; k++) {
        float s = z + _rz*sin(k*M_PI/180);
        float c = x + _rx*cos(k*M_PI/180);
        glVertex3f( c, y, s );
    }
    glEnd();
    glLineWidth(0.5f);
}


}} // namespace Tools::Selections
