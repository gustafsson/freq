#include "glprojection.h"
#include "gluunproject.h"
#include "gluinvertmatrix.h"
#include <string.h>
#include <algorithm>
#include <limits>
#include "printmatrix.h"
#include "exceptionassert.h"

glProjection::
        glProjection()
{
}


glProjecter::glProjecter(const glProjection& g)
    :
      viewport(g.viewport),
      mv_(g.modelview),
      p_(g.projection),
      p_inverse_(invert(g.projection))
{
    mv_inverse_ = invert(mv_);
#ifdef _DEBUG
    if (mv_[0][3] != 0 || mv_[1][3] != 0 || mv_[2][3] != 0 || mv_[3][3] != 1)
    {
        PRINTMATRIX(mv_);
        EXCEPTION_ASSERT(false);
    }
#endif
}


void glProjecter::
        computeUnitsPerPixel( vectord p, vectord::T& timePerPixel, vectord::T& scalePerPixel ) const
{
    // Find units per pixel at point 'p' with glUnProject
    vectord screen = project( p );
    vectord screenX=screen, screenY=screen;
    if (screen[0] > viewport[0] + viewport[2]/2)
        screenX[0]--;
    else
        screenX[0]++;

    if (screen[1] > viewport[1] + viewport[3]/2)
        screenY[1]--;
    else
        screenY[1]++;

    vectord
            wBase = unProject( screen ),
            w1 = unProject( screenX ),
            w2 = unProject( screenY );

    // Move out of the screen (towards the viewer along the 'zbuffer-axis' in screen coordinates)
    screen[2]-=1;
    screenX[2]-=1;
    screenY[2]-=1;

    // Calculate the ray cast direction for the pixel corresponding to 'p' as well as directions
    // for nearby pixels
    vectord
            dirBase = unProject( screen )-wBase,
            dir1 = unProject( screenX )-w1,
            dir2 = unProject( screenY )-w2;

    // A valid projection on the xz-plane exists if dir[1]>0 and wBase[1]>0
    vectord
            xzBase = wBase - dirBase*(wBase[1]/dirBase[1]),
            xz1 = w1 - dir1*(w1[1]/dir1[1]),
            xz2 = w2 - dir2*(w2[1]/dir2[1]);

    // compute {units in xz-plane} per {screen pixel}, that determines the required resolution
    // i.e How long along each axis on the heightmap project is one pixel in the x or y direction on the screen
    // If there is no valid intersection point the time per pixel is set to maximum distance which results
    // in a Heightmap::Reference that covers the entire signal.
    vectord::T timePerPixel_x = std::numeric_limits<vectord::T>::max ();
    vectord::T timePerPixel_y = std::numeric_limits<vectord::T>::max ();
    vectord::T scalePerPixel_x = std::numeric_limits<vectord::T>::max ();
    vectord::T scalePerPixel_y = std::numeric_limits<vectord::T>::max ();

    if (dir1[1] > 0 && dirBase[1] > 0) {
        timePerPixel_x = xz1[0]-xzBase[0];
        scalePerPixel_x = xz1[2]-xzBase[2];
    }
    if (dir2[1] > 0 && dirBase[1] > 0) {
        timePerPixel_y = xz2[0]-xzBase[0];
        scalePerPixel_y = xz2[2]-xzBase[2];
    }

    // time/freqPerPixel is how much difference in time/freq there can be when moving one pixel away from the
    // pixel that represents the closest point in ref
    timePerPixel = sqrt(timePerPixel_x*timePerPixel_x + timePerPixel_y*timePerPixel_y);
    scalePerPixel = sqrt(scalePerPixel_x*scalePerPixel_x + scalePerPixel_y*scalePerPixel_y);
}


vectord::T glProjecter::
        computePixelDistance( vectord p1, vectord p2 ) const
{
    vectord screen1 = project( p1 );
    vectord screen2 = project( p2 );
    screen1 -= screen2;
    return std::sqrt(screen1[0]*screen1[0]+screen1[1]*screen1[1]);
}


tvector<2,float> glProjecter::
        projectPartialDerivatives_xz( vectord p, tvector<2,double> d ) const
{
    vectord screen = project( p );
    vectord screen_x = project( p + vectord(d[0],0,0) );
    vectord screen_z = project( p + vectord(0,0,d[1]) );
    screen_x -= screen;
    screen_z -= screen;
    return tvector<2,float>{
        std::sqrt((float)(screen_x[0]*screen_x[0]+screen_x[1]*screen_x[1]))/d[0],
        std::sqrt((float)(screen_z[0]*screen_z[0]+screen_z[1]*screen_z[1]))/d[1]};
}


const matrixd& glProjecter::
        mvp() const
{
    if (!valid_mvp_)
    {
        mvp_ = p_ * mv_;
        valid_mvp_ = true;
    }
    return mvp_;
}


const matrixd& glProjecter::
        mvp_inverse() const
{
    if (!valid_mvp_inverse_)
    {
        mvp_inverse_ = mv_inverse_ * p_inverse_;
        valid_mvp_inverse_ = true;
    }
    return mvp_inverse_;
}


const matrixd& glProjecter::
        modelview() const
{
    return mv_;
}


const matrixd& glProjecter::
        modelview_inverse() const
{
    return mv_inverse_;
}


const matrixd& glProjecter::
        projection() const
{
    return p_;
}


const matrixd& glProjecter::
        projection_inverse() const
{
    return p_inverse_;
}


void glProjecter::translate(vectord x)
{
    {
        matrixd::T* v =  (matrixd::T*)&mv_;
        v[12] += v[0]*x[0] +
                 v[4]*x[1] +
                 v[8]*x[2];
        v[13] += v[1]*x[0] +
                 v[5]*x[1] +
                 v[9]*x[2];
        v[14] += v[2]*x[0] +
                 v[6]*x[1] +
                 v[10]*x[2];
        // assume v[3], v[7], v[11] are zero
        //v[15] += v[3]*x[0] +
        //         v[7]*x[1] +
        //         v[11]*x[2];
    }
    {
        matrixd::T* v =  (matrixd::T*)&mv_inverse_;
        // http://www.wolframalpha.com/input/?i=invert+%7B%7Bx_00%2Cx_01%2Cx_02%2Cx_03%7D%2C%7Bx_10%2Cx_11%2Cx_12%2Cx_13%7D%2C%7Bx_20%2Cx_21%2Cx_22%2Cx_23%7D%2C%7B0%2C0%2C0%2C1%7D%7D
        //for (int i=0;i<3;i++)
        //{
        //    v[i] -= v[3]*x[i];
        //    v[4+i] -= v[7]*x[i];
        //    v[8+i] -= v[11]*x[i];
        //    v[12+i] -= v[15]*x[i];
        //}
        // assume v[3], v[7], v[11] are zero, and v[15]==1
        v[12] -= x[0];
        v[13] -= x[1];
        v[14] -= x[2];
    }

    valid_mvp_inverse_ = valid_mvp_ = false;
}


void glProjecter::scale(vectord x)
{
    {
        matrixd::T* v = (matrixd::T*)&mv_;
        for (int i=0;i<3;i++)
        {
            v[i] *= x[0];
            v[4+i] *= x[1];
            v[8+i] *= x[2];
        }
    }
    {
        matrixd::T* v = (matrixd::T*)&mv_inverse_;

        for (int i=0;i<3;i++)
            x[i] = matrixd::T(1.)/x[i];

        for (int i=0;i<3;i++)
        {
            v[i] *= x[i];
            v[4+i] *= x[i];
            v[8+i] *= x[i];
            v[12+i] *= x[i];
        }
    }

    valid_mvp_inverse_ = valid_mvp_ = false;
}


void glProjecter::rotate(vectord axis, double rad)
{
    mv_ *= matrixd::rot (axis,rad);
    mv_inverse_ = matrixd::rot (axis,-rad) * mv_inverse_;
    valid_mvp_inverse_ = valid_mvp_ = false;
}


void glProjecter::
        mult(const matrixd& m, const matrixd& m_inverse)
{
    // glProjecter assumes that m[3],m[7],m[11]==0 and m[15]=1 in both mv_ and mv_inverse_.
#ifdef _DEBUG
    if (m[0][3] != 0 || m[1][3] != 0 || m[2][3] != 0 || m[3][3] != 1)
    {
        PRINTMATRIX(m);
        EXCEPTION_ASSERT(false);
    }
    if (m_inverse[0][3] != 0 || m_inverse[1][3] != 0 || m_inverse[2][3] != 0 || m_inverse[3][3] != 1)
    {
        PRINTMATRIX(m_inverse);
        EXCEPTION_ASSERT(false);
    }
#endif

    mv_ *= m;
    mv_inverse_ = m_inverse * mv_inverse_;
    valid_mvp_inverse_ = valid_mvp_ = false;
}


vectord glProjecter::
        camera() const
{
    // get camera position
    // https://www.opengl.org/discussion_boards/showthread.php/178484-Extracting-camera-position-from-a-ModelView-Matrix
    const auto& pos = mv_inverse_[3];
    return vectord{pos[0],pos[1],pos[2]};
}


vectord glProjecter::
        project(vectord obj, bool *r) const
{
    tvector<4,double> in(obj[0],obj[1],obj[2],1.0);
    in = mvp()*in;
    if (in[3]==0.0) { if(r)*r=false; return vectord(); }
    double iz = 1/in[3];
    in[0] = ((in[0]*iz)*0.5 + 0.5)* viewport[2] + viewport[0];
    in[1] = ((in[1]*iz)*0.5 + 0.5)* viewport[3] + viewport[1];
    in[2] = (in[2]*iz)*0.5 + 0.5;
//    in[0] /= in[3];
//    in[1] /= in[3];
//    in[2] /= in[3];
//    /* Map x, y and z to range 0-1 */
//    in[0] = in[0] * 0.5 + 0.5;
//    in[1] = in[1] * 0.5 + 0.5;
//    in[2] = in[2] * 0.5 + 0.5;

//    /* Map x,y to viewport */
//    in[0] = in[0] * viewport[2] + viewport[0];
//    in[1] = in[1] * viewport[3] + viewport[1];

    if(r)*r=true;
    return vectord(in[0],in[1],in[2]);
}


tvector<2,float> glProjecter::
        project2d(vectord obj) const
{
    vectord p = project(obj);
    return tvector<2,float>(p[0],p[1]);
}


vectord glProjecter::
        unProject(vectord win, bool *r) const
{
    tvector<4,double> in(win[0],win[1],win[2],1.0);

    /* Map x and y from window coordinates */
    in[0] = (in[0] - viewport[0]) / viewport[2];
    in[1] = (in[1] - viewport[1]) / viewport[3];

    /* Map to range -1 to 1 */
    in[0] = in[0] * 2 - 1;
    in[1] = in[1] * 2 - 1;
    in[2] = in[2] * 2 - 1;

    in = mvp_inverse()*in;
    if (in[3] == 0.0) {if(r)*r=false;return vectord();}
    in[0] /= in[3];
    in[1] /= in[3];
    in[2] /= in[3];
    if (r) *r = true;
    return vectord(in[0],in[1],in[2]);
}


#include "tmatrixstring.h"
#include "exceptionassert.h"

#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget


void glProjection::
        test()
{
    // It should describe how OpenGL object space is translated to screen space.
    {
        std::string name = "glProjection";
        int argc = 1;
        char * argv = &name[0];
        QApplication a(argc,&argv);
        QGLWidget w;
        w.makeCurrent ();

        glViewport (0,0,100,100);
        glProjection g;

#ifdef LEGACY_OPENGL
        glGetDoublev(GL_MODELVIEW_MATRIX, g.modelview.v ());
        glGetDoublev(GL_PROJECTION_MATRIX, g.projection.v ());
        glGetIntegerv(GL_VIEWPORT, g.viewport.v);

        double id4[]{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        int vp[]{0,0,100,100};

        EXCEPTION_ASSERT_EQUALS(g.modelview, (tmatrix<4, double>(id4)));
        EXCEPTION_ASSERT_EQUALS(g.projection, (tmatrix<4, double>(id4)));
        EXCEPTION_ASSERT_EQUALS(g.viewport, (tvector<4, int>(vp)));
#endif
    }
}
