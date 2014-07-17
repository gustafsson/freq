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

    GLmatrix& modelview();
    GLmatrix& projection();
    tvector<4,int>& viewport();

    const GLvector::T* modelview_matrix() const { return modelview_matrix_.v (); }
    const GLvector::T* projection_matrix() const { return projection_matrix_.v (); }
    const int* viewport_matrix() const { return viewport_matrix_.v; }

    GLvector gluProject(GLvector obj, bool *r=0) const;
    GLvector gluUnProject(GLvector win, bool *r=0) const;
    void computeUnitsPerPixel( GLvector p, GLvector::T& timePerPixel, GLvector::T& scalePerPixel ) const;
    GLvector::T computePixelDistance( GLvector p1, GLvector p2 ) const;

private:
    GLmatrix          modelview_matrix_;
    GLmatrix          projection_matrix_;
    tvector<4,int>    viewport_matrix_;

public:
    static void test();
};

#endif // GLPROJECTION_H
