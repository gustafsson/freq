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
        FrustumClip(const glProjection& gl_projection, float border_width, float border_height)
{
    update(gl_projection, border_width, border_height);
}

void NormalizePlane(tvector<4,GLfloat> & plane)
{
    float mag = sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
    plane = plane * (1.f/mag);
}

template<class T>
tvector<3,T> planeNormal(const tvector<4,T>& p) {
    return tvector<3,T>(p[0], p[1], p[2]);
}

void FrustumClip::
        update(const glProjection& gl_projection, float border_width, float border_height)
{
    // http://web.archive.org/web/20120531231005/http://crazyjoke.free.fr/doc/3D/plane%20extraction.pdf

    const tmatrix<4, GLfloat>& modelview = gl_projection.modelview;
    const tmatrix<4, GLfloat>& projection = gl_projection.projection;
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

    DEBUGLOG
    {
        NormalizePlane(left);
        NormalizePlane(right);
        NormalizePlane(top);
        NormalizePlane(bottom);
        NormalizePlane(near);
        NormalizePlane(far);
    }

    // get camera position
    // https://www.opengl.org/discussion_boards/showthread.php/178484-Extracting-camera-position-from-a-ModelView-Matrix
    {
        // Get plane normals
        GLvector n1 = planeNormal(M[0]);
        GLvector n2 = planeNormal(M[1]);
        GLvector n3 = planeNormal(M[2]);

        // Get plane distances
        float d1(M[0][3]);
        float d2(M[1][3]);
        float d3(M[2][3]);

        // Get the intersection of these 3 planes
        // http://paulbourke.net/geometry/3planes/
        GLvector n2n3 = n2 ^ n3;
        GLvector n3n1 = n3 ^ n1;
        GLvector n1n2 = n1 ^ n2;

        GLvector top = (n2n3 * d1) + (n3n1 * d2) + (n1n2 * d3);
        float denom = n1 % n2n3;

        camera = top * ( 1. / -denom);
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
        clipFrustum( vector<GLvector> l, GLvector* closest_i ) const
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

    if (closest_i)
        *closest_i = closestPointOnPoly(l, camera);

    return l;
}

vector<GLvector> FrustumClip::
        clipFrustum( GLvector corner[4], GLvector* closest_i ) const
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


std::vector<GLvector> FrustumClip::
        visibleXZ()
{
    auto rightMost = [](tvector<4,GLfloat> p1, tvector<4,GLfloat> p2)
    {
        float T = 0;
        tvector<4,GLfloat> planes [] = {p1, p2};
        for (tvector<4,GLfloat> plane : planes)
        {
            // skip parallell planes and planes pointing along the same direction
            if (1e-6 < -(planeNormal(plane) % GLvector(1,0,0)))
            {
                float s;
                GLvector br1 = planeIntersection (GLvector(0,0,0), GLvector(1,0,0), s, plane);
                if (0 < s) T = std::max(T, br1[0]);
                GLvector tr1 = planeIntersection (GLvector(0,0,1), GLvector(1,0,1), s, plane);
                if (0 < s) T = std::max(T, tr1[0]);
            }
        }
        return 0 == T ? std::numeric_limits<float>::max () : T;
    };

    float lr = rightMost(left, right);
    float tb = rightMost(top, bottom);
    float nf = rightMost(near, far);
    float T = std::min(std::min(lr, nf), tb);

    GLvector corner[4]=
    {
        GLvector( 0, 0, 0),
        GLvector( 0, 0, 1),
        GLvector( T, 0, 1),
        GLvector( T, 0, 0),
    };

    return clipFrustum (corner);
}


} // namespace Render
} // namespace Heightmap
