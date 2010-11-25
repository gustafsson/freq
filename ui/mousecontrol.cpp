#include "mousecontrol.h"
#include <math.h> // M_PI, sin

#include <tvector.h>

#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

namespace Ui
{

typedef tvector<3,GLdouble> GLvector;


template<typename f>
static GLvector gluProject(tvector<3,f> obj, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0) {
    GLvector win;
    bool s = (GLU_TRUE == gluProject(obj[0], obj[1], obj[2], model, proj, view, &win[0], &win[1], &win[2]));
    if(r) *r=s;
    return win;
}


template<typename f>
static GLvector gluUnProject(tvector<3,f> win, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0) {
    GLvector obj;
    bool s = (GLU_TRUE == ::gluUnProject(win[0], win[1], win[2], model, proj, view, &obj[0], &obj[1], &obj[2]));
    if(r) *r=s;
    return obj;
}


template<typename f>
static GLvector gluProject(tvector<3,f> obj, bool *r=0) {
    GLdouble model[16], proj[16];
    GLint view[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, view);
    return gluProject(obj, model, proj, view, r);
}


template<typename f>
static GLvector gluUnProject(tvector<3,f> win, bool *r=0) {
    GLdouble model[16], proj[16];
    GLint view[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, view);
    return gluUnProject(win, model, proj, view, r);
}


MouseControl::
        MouseControl()
            :
            down( false ),
            hold( 0 )
{}


float MouseControl::
        deltaX( float x )
{
    if( down )
        return x - lastx;

    return 0;
}


float MouseControl::
        deltaY( float y )
{
    if( down )
        return y - lasty;

    return 0;
}


bool MouseControl::
        worldPos(double &ox, double &oy, float scale)
{
    return worldPos(this->lastx, this->lasty, ox, oy, scale);
}


bool MouseControl::
        planePos(GLdouble x, GLdouble y, float &ox, float &oy, float scale)
{
    GLdouble dx, dy;
    bool r = worldPos(x, y, dx, dy, scale);
    ox = dx;
    oy = dy;
    return r;
}


bool MouseControl::
        worldPos(GLdouble x, GLdouble y, GLdouble &ox, GLdouble &oy, float scale)
{
    GLdouble s;
    bool test[2];
    GLvector win_coord, world_coord[2];

    win_coord = GLvector(x, y, 0.1);

    world_coord[0] = gluUnProject<GLdouble>(win_coord, &test[0]);
    //printf("CamPos1: %f: %f: %f\n", world_coord[0][0], world_coord[0][1], world_coord[0][2]);

    win_coord[2] = 0.6;
    world_coord[1] = gluUnProject<GLdouble>(win_coord, &test[1]);
    //printf("CamPos2: %f: %f: %f\n", world_coord[1][0], world_coord[1][1], world_coord[1][2]);

    s = (-world_coord[0][1]/(world_coord[1][1]-world_coord[0][1]));

    if (0==world_coord[1][1]-world_coord[0][1])
        s = 0;

    ox = world_coord[0][0] + s * (world_coord[1][0]-world_coord[0][0]);
    oy = world_coord[0][2] + s * (world_coord[1][2]-world_coord[0][2]);

    float minAngle = 3;
    if( s < 0 || world_coord[0][1]-world_coord[1][1] < scale*sin(minAngle *(M_PI/180)) * (world_coord[0]-world_coord[1]).length() )
        return false;

    return test[0] && test[1];
}


bool MouseControl::
        spacePos(double &out_x, double &out_y)
{
    return spacePos(this->lastx, this->lasty, out_x, out_y);
}


bool MouseControl::
        spacePos(double in_x, double in_y, double &out_x, double &out_y)
{
    bool test;
    GLvector win_coord, world_coord;

    win_coord = GLvector(in_x, in_y, 0.1);

    world_coord = gluUnProject<GLdouble>(win_coord, &test);
    out_x = world_coord[0];
    out_y = world_coord[2];
    return test;
}


void MouseControl::
        press( float x, float y )
{
    update( x, y );
    down = true;
}


void MouseControl::
        update( float x, float y )
{
    touch();
    lastx = x;
    lasty = y;
}


void MouseControl::
        release()
{
    //touch();
    down = false;
}


bool MouseControl::
        isTouched()
{
    if(hold == 0)
        return true;
    else
        return false;
}

} // namespace Ui
