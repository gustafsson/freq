// Need to include OpenGL headers in a specific order. So do it here first to
// make sure that the order is correct.

// TODO Tidy
#include "heightmap/reference.h"

#include "rendermodel.h"

// tool support
#include "tfr/cwtfilter.h"
#include "heightmap/collection.h"

#include "support/brushfilter.h"
#include "support/brushpaintkernel.h"
#include "support/toolglbrush.h"

// Sonic AWE
#include "tfr/filter.h"

#include "brushview.h"

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

namespace Tools {

BrushView::
        BrushView(BrushModel* model)
            :
            enabled( false ),
            gauss( ResamplePos(0,0), ResamplePos(0,0) ),
            model_( model )
{
}


BrushView::
        ~BrushView()
{

}


void BrushView::
        draw()
{
    if (enabled)
        drawCircle();
}


void BrushView::
        drawCircle()
{
    float
        x = gauss.pos.x,
        z = gauss.pos.y,
        _rx = gauss.sigma().x,
        _rz = gauss.sigma().y;
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

    glLineWidth(0.6f);
    glBegin(GL_LINE_LOOP);
    for (unsigned k=0; k<360; k++) {
        float s = z + _rz*sin(k*M_PI/180);
        float c = x + _rx*cos(k*M_PI/180);
        glVertex3f( c, y, s );
    }
    glEnd();
    glLineWidth(0.5f);
}

} // namespace Tools
