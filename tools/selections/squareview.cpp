#include "squareview.h"
#include "squaremodel.h"

#include <gl.h>
#include <TaskTimer.h>
#include <glPushContext.h>

namespace Tools { namespace Selections
{


SquareView::SquareView(SquareModel* model, Signal::Worker* worker)
    :
    enabled(false),
    visible(true),
    model_(model),
    worker_(worker)
{
}


SquareView::
        ~SquareView()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void SquareView::
        draw()
{
    if (visible)
        drawSelectionRectangle();
}


void SquareView::
        drawSelectionRectangle()
{
    float
        x1 = model_->a.time,
        z1 = model_->a.scale,
        x2 = model_->b.time,
        z2 = model_->b.scale;
    if (x1==x2 || z1==z2)
        return;

    float y = 1;

    glPushAttribContext ac;
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);
    glColor4f( 0, 0, 0, enabled ? .5 : 0.2);

    glBegin(GL_TRIANGLE_STRIP);
        glVertex3f( x1, 0, z1 );
        glVertex3f( x1, y, z1 );
        glVertex3f( x2, 0, z1 );
        glVertex3f( x2, y, z1 );
        glVertex3f( x2, 0, z2 );
        glVertex3f( x2, y, z2 );
        glVertex3f( x1, 0, z2 );
        glVertex3f( x1, y, z2 );
        glVertex3f( x1, 0, z1 );
        glVertex3f( x1, y, z1 );
    glEnd();

    glLineWidth(1.6f);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_LINE_LOOP);
        glVertex3f( x1, y, z1 );
        glVertex3f( x2, y, z1 );
        glVertex3f( x2, y, z2 );
        glVertex3f( x1, y, z2 );
    glEnd();
    glLineWidth(0.5f);
    glDepthMask(true);
}


void SquareView::
        drawSelectionRectangle2()
{
    float l = worker_->source()->length();
    glDepthMask(false);
    glColor4f( 0, 0, 0, enabled ? .5 : 0.2);
    float
            x1 = std::max(0.f, std::min(model_->a.time,  model_->b.time)),
            z1 = std::max(0.f, std::min(model_->a.scale, model_->b.scale)),
            x2 = std::min(l,   std::max(model_->a.time,  model_->b.time)),
            z2 = std::min(1.f, std::max(model_->a.scale, model_->b.scale));
    float y = 1;


    glBegin(GL_QUADS);
    glVertex3f( 0, y, 0 );
    glVertex3f( 0, y, 1 );
    glVertex3f( x1, y, 1 );
    glVertex3f( x1, y, 0 );

    glVertex3f( x1, y, 0 );
    glVertex3f( x2, y, 0 );
    glVertex3f( x2, y, z1 );
    glVertex3f( x1, y, z1 );

    glVertex3f( x1, y, 1 );
    glVertex3f( x2, y, 1 );
    glVertex3f( x2, y, z2 );
    glVertex3f( x1, y, z2 );

    glVertex3f( l, y, 0 );
    glVertex3f( l, y, 1 );
    glVertex3f( x2, y, 1 );
    glVertex3f( x2, y, 0 );


    if (x1>0) {
        glVertex3f( x1, y, z1 );
        glVertex3f( x1, 0, z1 );
        glVertex3f( x1, 0, z2 );
        glVertex3f( x1, y, z2 );
        glVertex3f( 0, y, 0 );
        glVertex3f( 0, 0, 0 );
        glVertex3f( 0, 0, 1 );
        glVertex3f( 0, y, 1 );
    } else {
        glVertex3f( 0, y, 0 );
        glVertex3f( 0, 0, 0 );
        glVertex3f( 0, 0, z1 );
        glVertex3f( 0, y, z1 );
        glVertex3f( 0, y, z2 );
        glVertex3f( 0, 0, z2 );
        glVertex3f( 0, 0, 1 );
        glVertex3f( 0, y, 1 );
    }

    if (x2<l) {
        glVertex3f( x2, y, z1 );
        glVertex3f( x2, 0, z1 );
        glVertex3f( x2, 0, z2 );
        glVertex3f( x2, y, z2 );
        glVertex3f( l, y, 0 );
        glVertex3f( l, 0, 0 );
        glVertex3f( l, 0, 1 );
        glVertex3f( l, y, 1 );
    } else {
        glVertex3f( l, y, 0 );
        glVertex3f( l, 0, 0 );
        glVertex3f( l, 0, z1 );
        glVertex3f( l, y, z1 );
        glVertex3f( l, y, z2 );
        glVertex3f( l, 0, z2 );
        glVertex3f( l, 0, 1 );
        glVertex3f( l, y, 1 );
    }

    if (z1>0) {
        glVertex3f( x1, y, z1 );
        glVertex3f( x1, 0, z1 );
        glVertex3f( x2, 0, z1 );
        glVertex3f( x2, y, z1 );
        glVertex3f( 0, y, 0 );
        glVertex3f( 0, 0, 0 );
        glVertex3f( l, 0, 0 );
        glVertex3f( l, y, 0 );
    } else {
        glVertex3f( 0, y, 0 );
        glVertex3f( 0, 0, 0 );
        glVertex3f( x1, 0, 0 );
        glVertex3f( x1, y, 0 );
        glVertex3f( x2, y, 0 );
        glVertex3f( x2, 0, 0 );
        glVertex3f( l, 0, 0 );
        glVertex3f( l, y, 0 );
    }

    if (z2<1) {
        glVertex3f( x1, y, z2 );
        glVertex3f( x1, 0, z2 );
        glVertex3f( x2, 0, z2 );
        glVertex3f( x2, y, z2 );
        glVertex3f( 0, y, 1 );
        glVertex3f( 0, 0, 1 );
        glVertex3f( l, 0, 1 );
        glVertex3f( l, y, 1 );
    } else {
        glVertex3f( 0, y, 1 );
        glVertex3f( 0, 0, 1 );
        glVertex3f( x1, 0, 1 );
        glVertex3f( x1, y, 1 );
        glVertex3f( x2, y, 1 );
        glVertex3f( x2, 0, 1 );
        glVertex3f( l, 0, 1 );
        glVertex3f( l, y, 1 );
    }
    glEnd();

    glDepthMask(true);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_QUADS);
    if (x1>0) {
        glVertex3f( x1, y, z1 );
        glVertex3f( x1, 0, z1 );
        glVertex3f( x1, 0, z2 );
        glVertex3f( x1, y, z2 );
    }

    if (x2<l) {
        glVertex3f( x2, y, z1 );
        glVertex3f( x2, 0, z1 );
        glVertex3f( x2, 0, z2 );
        glVertex3f( x2, y, z2 );
    }

    if (z1>0) {
        glVertex3f( x1, y, z1 );
        glVertex3f( x1, 0, z1 );
        glVertex3f( x2, 0, z1 );
        glVertex3f( x2, y, z1 );
    }

    if (z2<1) {
        glVertex3f( x1, y, z2 );
        glVertex3f( x1, 0, z2 );
        glVertex3f( x2, 0, z2 );
        glVertex3f( x2, y, z2 );
    }
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}


}} // namespace Tools::Selections
