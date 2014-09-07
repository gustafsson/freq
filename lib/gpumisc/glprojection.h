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

    matrixd           modelview;
    matrixd           projection;
    tvector<4,int>    viewport;

    vectord gluProject(vectord obj, bool *r=0) const;
    vectord gluUnProject(vectord win, bool *r=0) const;
    void computeUnitsPerPixel( vectord p, vectord::T& timePerPixel, vectord::T& scalePerPixel ) const;
    vectord::T computePixelDistance( vectord p1, vectord p2 ) const;

public:
    static void test();
};

#endif // GLPROJECTION_H
