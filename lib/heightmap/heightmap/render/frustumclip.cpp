#include "frustumclip.h"

// gpusmisc
#include "gluunproject.h"
#include "geometricalgebra.h"

#include <stdio.h>

//#define DEBUGLOG
#define DEBUGLOG if(0)

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

void NormalizePlane(tvector<4,GLfloat> & plane)
{
    float mag = sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
    plane = plane * (1.f/mag);
}

void FrustumClip::
        update(float border_width, float border_height)
{
    // http://web.archive.org/web/20120531231005/http://crazyjoke.free.fr/doc/3D/plane%20extraction.pdf

    tmatrix<4, GLfloat> modelview = gl_projection->modelview_matrix ();
    tmatrix<4, GLfloat> projection = gl_projection->projection_matrix ();
    tmatrix<4, GLfloat> M = projection*modelview;
    M = M.transpose ();
    border_width = 1.f + 2.f*border_width;
    border_height = 1.f + 2.f*border_height;
    left   = M[3] + M[0]*border_width;
    right  = M[3] - M[0]*border_width;
    top    = M[3] - M[1]*border_height;
    bottom = M[3] + M[1]*border_height;
    near   = M[3] + M[2];
    far    = M[3] - M[2];

    // must make all normals negative because one of the axes is flipped (glScale with a minus sign on the x-axis)
    right[3] = -right[3];
    left[3] = -left[3];
    top[3] = -top[3];
    bottom[3] = -bottom[3];

    DEBUGLOG
    {
        NormalizePlane(left);
        NormalizePlane(right);
        NormalizePlane(top);
        NormalizePlane(bottom);
        NormalizePlane(near);
        NormalizePlane(far);
    }
}

#define PRINTL(P,L) printl(#P, P, L)

inline void printl(const char* str, tvector<4,GLfloat> n, const std::vector<GLvector>& l) {
    fprintf(stdout,"%s: %.2f, %.2f, %.2f, %.2f\n",str,n[0],n[1],n[2],n[3]);
    for (unsigned i=0; i<l.size(); i++)
        fprintf(stdout,"  %.2f, %.2f, %.2f\n",l[i][0],l[i][1],l[i][2]);
    fflush(stdout);
}

vector<GLvector> FrustumClip::
        clipFrustum( vector<GLvector> l, GLvector &closest_i ) const
{
    DEBUGLOG printl("Start", tvector<4,GLfloat>(), l);
    l = clipPlane(l, right);
    DEBUGLOG PRINTL(right, l);
    l = clipPlane(l, left);
    DEBUGLOG PRINTL(left, l);
    l = clipPlane(l, top);
    DEBUGLOG PRINTL(top,l);
    l = clipPlane(l, bottom);
    DEBUGLOG PRINTL(bottom,l);

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
