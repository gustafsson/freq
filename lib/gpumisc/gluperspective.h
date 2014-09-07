#ifndef GLUPERSPECTIVE_H
#define GLUPERSPECTIVE_H

void glhPerspective(double *matrix, double fovyInDegrees, double aspectRatio,
                      double znear, double zfar);
void glhFrustum(double *matrix, double left, double right, double bottom, double top,
                  double znear, double zfar);
void glhOrtho(double *matrix, double left, double right, double bottom, double top,
                  double znear, double zfar);

#endif // GLUPERSPECTIVE_H
