#include "brushview.h"

#ifdef _MSC_VER // gl.h expects windows.h to be included on windows
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <GL/gl.h>
#include <glPushContext.h>

namespace Tools {

BrushView::
        BrushView(BrushModel* model)
            :
            enabled( false ),
            gauss( make_float2(0,0), make_float2(0,0) ),
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

    glLineWidth(0.6f);
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

} // namespace Tools
