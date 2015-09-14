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

public:
    static void test();
};


class glProjecter
{
public:
    glProjecter(const glProjection& p);
    const matrixd& mvp() const;
    const matrixd& mvp_inverse() const;
    const matrixd& modelview() const;
    const matrixd& modelview_inverse() const;
    const matrixd& projection() const;
    const matrixd& projection_inverse() const;
    const tvector<4,int> viewport;

    void translate(vectord x);
    void scale(vectord x);
    void rotate(vectord axis, double rad);
    void mult(const matrixd& m, const matrixd& m_inverse);

    vectord project(vectord obj, bool *r=0) const;
    vectord unProject(vectord win, bool *r=0) const;
    void computeUnitsPerPixel( vectord p, vectord::T& timePerPixel, vectord::T& scalePerPixel ) const;
    vectord::T computePixelDistance( vectord p1, vectord p2 ) const;
private:
    bool mutable valid_mvp_=false;
    bool mutable valid_mvp_inverse_=false;
    mutable matrixd mvp_;
    mutable matrixd mvp_inverse_;
    matrixd mv_;
    matrixd mv_inverse_;
    const matrixd p_;
    const matrixd p_inverse_;
};


#endif // GLPROJECTION_H
