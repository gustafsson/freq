#include "frustumclip.h"

// gpusmisc
#include "gluunproject.h"
#include "geometricalgebra.h"

#include <stdio.h>

using namespace std;
using namespace GeometricAlgebra;

namespace Heightmap {
namespace Render {

FrustumClip::
        FrustumClip(glProjection* gl_projection, bool* left_handed_axes)
    :
      gl_projection(gl_projection),
      left_handed_axes(left_handed_axes)
{
}


void FrustumClip::
        update(float w, float h)
{
    // this takes about 5 us
    GLint const* const& view = gl_projection->viewport_matrix ();
    glProjection* p = gl_projection;

    double z0=.1, z1=.2;

    projectionPlane = p->gluUnProject( GLvector( view[0] + view[2]/2, view[1] + view[3]/2, z0) );
    projectionNormal = (p->gluUnProject( GLvector( view[0] + view[2]/2, view[1] + view[3]/2, z1) ) - projectionPlane).Normalized();

    rightPlane = p->gluUnProject( GLvector( view[0] + (1-w)*view[2], view[1] + view[3]/2, z0) );
    GLvector rightZ = p->gluUnProject( GLvector( view[0] + (1-w)*view[2], view[1] + view[3]/2, z1) );
    GLvector rightY = p->gluUnProject( GLvector( view[0] + (1-w)*view[2], view[1] + view[3]/2+1, z0) );
    rightZ = rightZ - rightPlane;
    rightY = rightY - rightPlane;
    rightNormal = ((rightY)^(rightZ)).Normalized();

    leftPlane = p->gluUnProject( GLvector( view[0]+w*view[2], view[1] + view[3]/2, z0) );
    GLvector leftZ = p->gluUnProject( GLvector( view[0]+w*view[2], view[1] + view[3]/2, z1) );
    GLvector leftY = p->gluUnProject( GLvector( view[0]+w*view[2], view[1] + view[3]/2+1, z0) );
    leftNormal = ((leftZ-leftPlane)^(leftY-leftPlane)).Normalized();

    topPlane = p->gluUnProject( GLvector( view[0] + view[2]/2, view[1] + (1-h)*view[3], z0) );
    GLvector topZ = p->gluUnProject( GLvector( view[0] + view[2]/2, view[1] + (1-h)*view[3], z1) );
    GLvector topX = p->gluUnProject( GLvector( view[0] + view[2]/2+1, view[1] + (1-h)*view[3], z0) );
    topNormal = ((topZ-topPlane)^(topX-topPlane)).Normalized();

    bottomPlane = p->gluUnProject( GLvector( view[0] + view[2]/2, view[1]+h*view[3], z0) );
    GLvector bottomZ = p->gluUnProject( GLvector( view[0] + view[2]/2, view[1]+h*view[3], z1) );
    GLvector bottomX = p->gluUnProject( GLvector( view[0] + view[2]/2+1, view[1]+h*view[3], z0) );
    bottomNormal = ((bottomX-bottomPlane)^(bottomZ-bottomPlane)).Normalized();

    // must make all normals negative because one of the axes is flipped (glScale with a minus sign on the x-axis)
    if (*left_handed_axes)
    {
        rightNormal = -rightNormal;
        leftNormal = -leftNormal;
        topNormal = -topNormal;
        bottomNormal = -bottomNormal;
    }

    // Don't bother with projectionNormal?
    projectionNormal = projectionNormal;
}


inline void printl(const char* str, const std::vector<GLvector>& l) {
    fprintf(stdout,"%s (%lu)\n",str,(unsigned long)l.size());
    for (unsigned i=0; i<l.size(); i++) {
        fprintf(stdout,"  %g\t%g\t%g\n",l[i][0],l[i][1],l[i][2]);
    }
    fflush(stdout);
}


vector<GLvector> FrustumClip::
        clipFrustum( vector<GLvector> l, GLvector &closest_i ) const
{
    //printl("Start",l);
    // Don't bother with projectionNormal?
    //clipPlane(l, projectionPlane, projectionNormal);
    //printl("Projectionclipped",l);
    l = clipPlane(l, rightPlane, rightNormal);
    //printl("Right", l);
    l = clipPlane(l, leftPlane, leftNormal);
    //printl("Left", l);
    l = clipPlane(l, topPlane, topNormal);
    //printl("Top",l);
    l = clipPlane(l, bottomPlane, bottomNormal);
    //printl("Bottom",l);
    //printl("Clipped polygon",l);

    closest_i = closestPointOnPoly(l, projectionPlane);
    return l;
}


vector<GLvector> FrustumClip::
        clipFrustum( GLvector corner[4], GLvector &closest_i ) const
{
    vector<GLvector> l;
    l.reserve(4);
    for (unsigned i=0; i<4; i++)
    {
        if (!l.empty() && l.back() == corner[i])
            continue;

        l.push_back( corner[i] );
    }

    return clipFrustum(l, closest_i );
}


} // namespace Render
} // namespace Heightmap
