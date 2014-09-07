/**

  http://www.codng.com/2011/02/gluunproject-for-iphoneios.html

  */


#ifndef GLUPROJECT_IOS_H
#define GLUPROJECT_IOS_H

#include "gl.h"
#include "GLvector.h"
//#include <OpenGLES/ES1/gl.h>
//#include <OpenGLES/ES1/glext.h>

matrixd
gluPerspective(double fovy, double aspect, double zNear, double zFar);

matrixd
gluLookAt(double eyex, double eyey, double eyez, double centerx,
    double centery, double centerz, double upx, double upy,
    double upz);

GLint
gluProject(double objx, double objy, double objz,
     const double modelMatrix[16],
     const double projMatrix[16],
     const GLint viewport[4],
     double *winx, double *winy, double *winz);

GLint
gluUnProject(double winx, double winy, double winz,
    const double modelMatrix[16],
    const double projMatrix[16],
    const GLint viewport[4],
    double *objx, double *objy, double *objz);


GLint
gluUnProject4(double winx, double winy, double winz, double clipw,
     const double modelMatrix[16],
     const double projMatrix[16],
     const GLint viewport[4],
     GLclampf nearVal, GLclampf farVal,
     double *objx, double *objy, double *objz,
     double *objw);

matrixd
gluPickMatrix(double x, double y, double deltax, double deltay,
     GLint viewport[4]);

#endif // GLUPROJECT_IOS_H
