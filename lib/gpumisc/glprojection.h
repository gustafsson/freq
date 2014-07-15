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

    void update(GLvector::T modelview_matrix[16],
                GLvector::T projection_matrix[16],
                int viewport_matrix[4]);

    // scales computeUnitsPerPixel
    void setZoom(float zoom);
    float getZoom();

    const GLvector::T* modelview_matrix() const { return modelview_matrix_; }
    const GLvector::T* projection_matrix() const { return projection_matrix_; }
    const int* viewport_matrix() const { return viewport_matrix_; }

    GLvector gluProject(GLvector obj, bool *r=0);
    GLvector gluUnProject(GLvector win, bool *r=0);
    void computeUnitsPerPixel( GLvector p, GLvector::T& timePerPixel, GLvector::T& scalePerPixel );
    GLvector::T computePixelDistance( GLvector p1, GLvector p2 );

private:
    float                           zoom;
    GLvector::T                     modelview_matrix_[16];
    GLvector::T                     projection_matrix_[16];
    int                             viewport_matrix_[4];

public:
    static void test();
};

#endif // GLPROJECTION_H
