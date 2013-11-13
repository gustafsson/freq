#ifndef GLPROJECTION_H
#define GLPROJECTION_H

#include "GLvector.h"

/**
 * @brief The glProjection class should describe how OpenGL object space is
 * translated to screen space.
 */
class glProjection
{
public:
    glProjection();

    void update();

    const double* modelview_matrix() const { return modelview_matrix_; }
    const double* projection_matrix() const { return projection_matrix_; }
    const int* viewport_matrix() const { return viewport_matrix_; }

    GLvector gluProject(GLvector obj, bool *r=0);
    GLvector gluUnProject(GLvector win, bool *r=0);
    void computeUnitsPerPixel( GLvector p, GLvector::T& timePerPixel, GLvector::T& scalePerPixel );

private:
    double                          modelview_matrix_[16];
    double                          projection_matrix_[16];
    int                             viewport_matrix_[4];

public:
    static void test();
};

#endif // GLPROJECTION_H
