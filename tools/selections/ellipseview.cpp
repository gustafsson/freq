#include "ellipseview.h"
#include "ellipsemodel.h"

#include <GL/gl.h>
#include <TaskTimer.h>
#include <glPushContext.h>

namespace Tools { namespace Selections
{


EllipseView::EllipseView(EllipseModel* model)
    :
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
    drawSelectionCircle();
}

void EllipseView::
        drawSelectionCircle()
{
    float
        x = model_->a.time,
        z = model_->a.scale,
        _rx = fabs(model_->b.time - model_->a.time),
        _rz = fabs(model_->b.scale - model_->a.scale);
    float y = 1;

    glPushAttribContext ac;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);
    glColor4f( 0, 0, 0, enabled ? .5 : 0.2);

    glBegin(GL_TRIANGLE_STRIP);
    for (unsigned k=0; k<=360; k++) {
        float s = z + _rz*sin(k*M_PI/180);
        float c = x + _rx*cos(k*M_PI/180);
        glVertex3f( c, 0, s );
        glVertex3f( c, y, s );
    }
    glEnd();

    glLineWidth(3.2f);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_LINE_LOOP);
    for (unsigned k=0; k<360; k++) {
        float s = z + _rz*sin(k*M_PI/180);
        float c = x + _rx*cos(k*M_PI/180);
        glVertex3f( c, y, s );
    }
    glEnd();
    glLineWidth(0.5f);
    glDepthMask(true);
}


}} // namespace Selections::Tools
