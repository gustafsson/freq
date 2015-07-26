#include "glprojection.h"
#include "gluunproject.h"
#include <string.h>
#include <algorithm>
#include <limits>

glProjection::
        glProjection()
{
}


vectord glProjection::
        gluProject(vectord obj, bool *r) const
{
    return ::gluProject(obj, modelview.v (), projection.v (), viewport.v, r);
}


vectord glProjection::
        gluUnProject(vectord win, bool *r) const
{
    return ::gluUnProject(win, modelview.v (), projection.v (), viewport.v, r);
}


void glProjection::
        computeUnitsPerPixel( vectord p, vectord::T& timePerPixel, vectord::T& scalePerPixel ) const
{
    // Find units per pixel at point 'p' with glUnProject
    vectord screen = gluProject( p );
    vectord screenX=screen, screenY=screen;
    if (screen[0] > viewport[0] + viewport[2]/2)
        screenX[0]--;
    else
        screenX[0]++;

    if (screen[1] > viewport[1] + viewport[3]/2)
        screenY[1]--;
    else
        screenY[1]++;

    vectord
            wBase = gluUnProject( screen ),
            w1 = gluUnProject( screenX ),
            w2 = gluUnProject( screenY );

    // Move out of the screen (towards the viewer along the 'zbuffer-axis' in screen coordinates)
    screen[2]-=1;
    screenX[2]-=1;
    screenY[2]-=1;

    // Calculate the ray cast direction for the pixel corresponding to 'p' as well as directions
    // for nearby pixels
    vectord
            dirBase = gluUnProject( screen )-wBase,
            dir1 = gluUnProject( screenX )-w1,
            dir2 = gluUnProject( screenY )-w2;

    // A valid projection on the xz-plane exists if dir[1]>0 and wBase[1]>0
    vectord
            xzBase = wBase - dirBase*(wBase[1]/dirBase[1]),
            xz1 = w1 - dir1*(w1[1]/dir1[1]),
            xz2 = w2 - dir2*(w2[1]/dir2[1]);

    // compute {units in xz-plane} per {screen pixel}, that determines the required resolution
    // i.e How long along each axis on the heightmap project is one pixel in the x or y direction on the screen
    // If there is no valid intersection point the time per pixel is set to maximum distance which results
    // in a Heightmap::Reference that covers the entire signal.
    vectord::T timePerPixel_x = std::numeric_limits<vectord::T>::max ();
    vectord::T timePerPixel_y = std::numeric_limits<vectord::T>::max ();
    vectord::T scalePerPixel_x = std::numeric_limits<vectord::T>::max ();
    vectord::T scalePerPixel_y = std::numeric_limits<vectord::T>::max ();

    if (dir1[1] > 0 && dirBase[1] > 0) {
        timePerPixel_x = xz1[0]-xzBase[0];
        scalePerPixel_x = xz1[2]-xzBase[2];
    }
    if (dir2[1] > 0 && dirBase[1] > 0) {
        timePerPixel_y = xz2[0]-xzBase[0];
        scalePerPixel_y = xz2[2]-xzBase[2];
    }

    // time/freqPerPixel is how much difference in time/freq there can be when moving one pixel away from the
    // pixel that represents the closest point in ref
    timePerPixel = sqrt(timePerPixel_x*timePerPixel_x + timePerPixel_y*timePerPixel_y);
    scalePerPixel = sqrt(scalePerPixel_x*scalePerPixel_x + scalePerPixel_y*scalePerPixel_y);
}


vectord::T glProjection::
        computePixelDistance( vectord p1, vectord p2 ) const
{
    vectord screen1 = gluProject( p1 );
    vectord screen2 = gluProject( p2 );
    return (screen2-screen1).length();
}


#include "tmatrixstring.h"
#include "exceptionassert.h"

#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget


void glProjection::
        test()
{
    // It should describe how OpenGL object space is translated to screen space.
    {
        std::string name = "glProjection";
        int argc = 1;
        char * argv = &name[0];
        QApplication a(argc,&argv);
        QGLWidget w;
        w.makeCurrent ();

        glViewport (0,0,100,100);
        glProjection g;

#ifdef LEGACY_OPENGL
        glGetDoublev(GL_MODELVIEW_MATRIX, g.modelview.v ());
        glGetDoublev(GL_PROJECTION_MATRIX, g.projection.v ());
        glGetIntegerv(GL_VIEWPORT, g.viewport.v);

        double id4[]{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        int vp[]{0,0,100,100};

        EXCEPTION_ASSERT_EQUALS(g.modelview, (tmatrix<4, double>(id4)));
        EXCEPTION_ASSERT_EQUALS(g.projection, (tmatrix<4, double>(id4)));
        EXCEPTION_ASSERT_EQUALS(g.viewport, (tvector<4, int>(vp)));
#endif
    }
}
