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


class glProjectionProjecter
{
public:
    glProjectionProjecter(const matrixd& mvp, const tvector<4,int>& viewport);
    const matrixd& mvp() const {return mvp_;}
    const matrixd& mvp_inverse() const {return mvp_inverse_;}
    const tvector<4,int>& viewport() const {return viewport_;}

    void translate(vectord x);
    void scale(vectord x);
    void rotate(vectord axis, double rad);
    void mult(matrixd& m);

    vectord project(vectord obj, bool *r=0) const;
    vectord unProject(vectord win, bool *r=0) const;
    void computeUnitsPerPixel( vectord p, vectord::T& timePerPixel, vectord::T& scalePerPixel ) const;
    vectord::T computePixelDistance( vectord p1, vectord p2 ) const;
private:
    tvector<4,int> viewport_;
    matrixd mvp_;
    matrixd mvp_inverse_;
};


#endif // GLPROJECTION_H
