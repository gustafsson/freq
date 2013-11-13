#include "glprojection.h"
#include "gluunproject.h"
#include <string.h>
#include <algorithm>

glProjection::
        glProjection()
{
    memset(modelview_matrix_, 0, sizeof(modelview_matrix_));
    memset(projection_matrix_, 0, sizeof(projection_matrix_));
    memset(viewport_matrix_, 0, sizeof(viewport_matrix_));

    update();
}


void glProjection::
        update()
{
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix_);
    glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix_);
    glGetIntegerv(GL_VIEWPORT, viewport_matrix_);
}


GLvector glProjection::
        gluProject(GLvector obj, bool *r)
{
    return ::gluProject(obj, modelview_matrix_, projection_matrix_, viewport_matrix_, r);
}


GLvector glProjection::
        gluUnProject(GLvector win, bool *r)
{
    return ::gluUnProject(win, modelview_matrix_, projection_matrix_, viewport_matrix_, r);
}


void glProjection::
        computeUnitsPerPixel( GLvector p, GLvector::T& timePerPixel, GLvector::T& scalePerPixel )
{
    // Find units per pixel at point 'p' with glUnProject
    GLvector screen = gluProject( p );
    GLvector screenX=screen, screenY=screen;
    if (screen[0] > viewport_matrix_[0] + viewport_matrix_[2]/2)
        screenX[0]--;
    else
        screenX[0]++;

    if (screen[1] > viewport_matrix_[1] + viewport_matrix_[3]/2)
        screenY[1]--;
    else
        screenY[1]++;

    GLvector
            wBase = gluUnProject( screen ),
            w1 = gluUnProject( screenX ),
            w2 = gluUnProject( screenY );

    screen[2]+=1;
    screenX[2]+=1;
    screenY[2]+=1;

    // directions
    GLvector
            dirBase = gluUnProject( screen )-wBase,
            dir1 = gluUnProject( screenX )-w1,
            dir2 = gluUnProject( screenY )-w2;

    // valid projection on xz-plane exists if dir?[1]<0 wBase[1]<0
    GLvector
            xzBase = wBase - dirBase*(wBase[1]/dirBase[1]),
            xz1 = w1 - dir1*(w1[1]/dir1[1]),
            xz2 = w2 - dir2*(w2[1]/dir2[1]);

    // compute {units in xz-plane} per {screen pixel}, that determines the required resolution
    timePerPixel = 0;
    scalePerPixel = 0;

    if (dir1[1] != 0 && dirBase[1] != 0) {
        timePerPixel = std::max(timePerPixel, fabs(xz1[0]-xzBase[0]));
        scalePerPixel = std::max(scalePerPixel, fabs(xz1[2]-xzBase[2]));
    }
    if (dir2[1] != 0 && dirBase[1] != 0) {
        timePerPixel = std::max(timePerPixel, fabs(xz2[0]-xzBase[0]));
        scalePerPixel = std::max(scalePerPixel, fabs(xz2[2]-xzBase[2]));
    }

    if (0 == timePerPixel)
        timePerPixel = std::max(fabs(w1[0]-wBase[0]), fabs(w2[0]-wBase[0]));
    if (0 == scalePerPixel)
        scalePerPixel = std::max(fabs(w1[2]-wBase[2]), fabs(w2[2]-wBase[2]));

    if (0==scalePerPixel) scalePerPixel=timePerPixel;
    if (0==timePerPixel) timePerPixel=scalePerPixel;

    // time/freqPerPixel is how much difference in time/freq there can be when moving one pixel away from the
    // pixel that represents the closest point in ref
}


#include "tmatrixstring.h"
#include "exceptionassert.h"

#include <QApplication>
#include <QGLWidget>


void glProjection::
        test()
{
    // It should describe how OpenGL object space is translated to screen space.
    {
        int argc = 0;
        char* argv = 0;
        QApplication a(argc,&argv);
        QGLWidget w;
        w.makeCurrent ();

        glViewport (0,0,100,100);
        glProjection g;

        tmatrix<4, double> modelview = g.modelview_matrix ();
        tmatrix<4, double> projection = g.projection_matrix ();
        tvector<4, int> viewport = g.viewport_matrix ();

        double id4[]{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        double vp[]{0,0,100,100};

        EXCEPTION_ASSERT_EQUALS(modelview, (tmatrix<4, double>(id4)));
        EXCEPTION_ASSERT_EQUALS(projection, (tmatrix<4, double>(id4)));
        EXCEPTION_ASSERT_EQUALS(viewport, (tvector<4, double>(vp)));
    }
}
