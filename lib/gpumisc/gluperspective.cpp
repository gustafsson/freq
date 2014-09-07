// http://www.opengl.org/wiki/GluPerspective_code

#include "gluperspective.h"
#include <cmath>
#include <string.h>

//matrix will receive the calculated perspective matrix.
//You would have to upload to your shader
// or use glLoadMatrixf if you aren't using shaders.
void glhPerspective(double *matrix, double fovyInDegrees, double aspectRatio,
                      double znear, double zfar)
{
    double ymax, xmax;
    ymax = znear * tanf(fovyInDegrees * M_PI / 360.0);
    //ymin = -ymax;
    //xmin = -ymax * aspectRatio;
    xmax = ymax * aspectRatio;
    glhFrustum(matrix, -xmax, xmax, -ymax, ymax, znear, zfar);
}


void glhFrustum(double *matrix, double left, double right, double bottom, double top,
                  double znear, double zfar)
{
    double temp, temp2, temp3, temp4;
    temp = 2.0 * znear;
    temp2 = right - left;
    temp3 = top - bottom;
    temp4 = zfar - znear;
    matrix[0] = temp / temp2;
    matrix[1] = 0.0;
    matrix[2] = 0.0;
    matrix[3] = 0.0;
    matrix[4] = 0.0;
    matrix[5] = temp / temp3;
    matrix[6] = 0.0;
    matrix[7] = 0.0;
    matrix[8] = (right + left) / temp2;
    matrix[9] = (top + bottom) / temp3;
    matrix[10] = (-zfar - znear) / temp4;
    matrix[11] = -1.0;
    matrix[12] = 0.0;
    matrix[13] = 0.0;
    matrix[14] = (-temp * zfar) / temp4;
    matrix[15] = 0.0;
}


void glhOrtho(double *matrix, double left, double right, double bottom, double top,
                  double near, double far)
{
    double a = 2.0f / (right - left);
    double b = 2.0f / (top - bottom);
    double c = -2.0f / (far - near);

    double tx = - (right + left)/(right - left);
    double ty = - (top + bottom)/(top - bottom);
    double tz = - (far + near)/(far - near);

    double ortho[16] = {
        a, 0, 0, 0,
        0, b, 0, 0,
        0, 0, c, 0,
        tx, ty, tz, 1
    };

    memcpy(matrix, ortho, sizeof(ortho));
}
