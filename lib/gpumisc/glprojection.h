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

    GLmatrix          modelview;
    GLmatrix          projection;
    tvector<4,int>    viewport;

    GLvector gluProject(GLvector obj, bool *r=0) const;
    GLvector gluUnProject(GLvector win, bool *r=0) const;
    void computeUnitsPerPixel( GLvector p, GLvector::T& timePerPixel, GLvector::T& scalePerPixel ) const;
    GLvector::T computePixelDistance( GLvector p1, GLvector p2 ) const;

public:
    static void test();
};

#endif // GLPROJECTION_H
