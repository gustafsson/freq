#ifndef GLPROJECTION_H
#define GLPROJECTION_H

#include "GLvector.h"
#include "tmatrix.h"

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

    const GLvector::T* modelview_matrix() const { return modelview_matrix_.v (); }
    const GLvector::T* projection_matrix() const { return projection_matrix_.v (); }
    const int* viewport_matrix() const { return viewport_matrix_.v; }

    GLvector gluProject(GLvector obj, bool *r=0);
    GLvector gluUnProject(GLvector win, bool *r=0);
    void computeUnitsPerPixel( GLvector p, GLvector::T& timePerPixel, GLvector::T& scalePerPixel );
    GLvector::T computePixelDistance( GLvector p1, GLvector p2 );

private:
    float                           zoom;
    tmatrix<4,GLvector::T>          modelview_matrix_;
    tmatrix<4,GLvector::T>          projection_matrix_;
    tvector<4,int>                  viewport_matrix_;

public:
    static void test();
};

#endif // GLPROJECTION_H
