#ifndef GLUPERSPECTIVE_H
#define GLUPERSPECTIVE_H

void glhPerspectivef(float *matrix, float fovyInDegrees, float aspectRatio,
                      float znear, float zfar);
void glhFrustumf(float *matrix, float left, float right, float bottom, float top,
                  float znear, float zfar);
void glhOrtho(float *matrix, float left, float right, float bottom, float top,
                  float znear, float zfar);

#endif // GLUPERSPECTIVE_H
