#include "frustumclip.h"

// gpusmisc
#include "gluunproject.h"

#include <cmath>
#include <float.h>

using namespace std;

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


/* distance along normal, a negative distance means obj is in front of plane */
static float distanceToPlane( GLvector obj, const GLvector& plane, const GLvector& normal ) {
    return (plane-obj)%normal;
}


/* returns the point on the border of the polygon 'l' that lies closest to 'target' */
static GLvector closestPointOnPoly( const std::vector<GLvector>& l, const GLvector &target)
{
    GLvector r;
    // check if point lies inside
    bool allNeg = true, allPos = true;

    // find point in poly closest to projectionPlane
    float min = FLT_MAX;
    for (unsigned i=0; i<l.size(); i++) {
        float f = (l[i]-target).dot();
        if (f<min) {
            min = f;
            r = l[i];
        }

        GLvector d = (l[(i+1)%l.size()] - l[i]),
                 v = target - l[i];

        if (0==d.dot())
            continue;

        if (d%v < 0) allNeg=false;
        if (d%v > 0) allPos=false;
        float k = d%v / (d.dot());
        if (0<k && k<1) {
            f = (l[i] + d*k-target).dot();
            if (f<min) {
                min = f;
                r = l[i]+d*k;
            }
        }
    }

    if (allNeg || allPos) {
        // point lies within convex polygon, create normal and project to surface
        if (l.size()>2) {
            GLvector n = (l[0]-l[1])^(l[0]-l[2]);
            if (0 != n.dot()) {
                n = n.Normalized();
                r = target + n*distanceToPlane( target, l[0], n );
            }
        }
    }
    return r;
}


// the normal does not need to be normalized
static GLvector planeIntersection( GLvector const& pt1, GLvector const& pt2, float &s, GLvector const& plane, GLvector const& normal ) {
    GLvector dir = pt2-pt1;

    s = ((plane-pt1)%normal)/(dir % normal);
    GLvector p = pt1 + dir * s;

//    float v = (p-plane ) % normal;
//    fprintf(stdout,"p[2] = %g, v = %g\n", p[2], v);
//    fflush(stdout);
    return p;
}


static void clipPlane( vector<GLvector>& p, const GLvector& p0, const GLvector& n )
{
    if (p.empty())
        return;

    unsigned i;

    GLvector const* a, * b = &p[p.size()-1];
    bool a_side, b_side = (p0-*b)%n < 0;
    for (i=0; i<p.size(); i++) {
        a = b;
        b = &p[i];

        a_side = b_side;
        b_side = (p0-*b)%n < 0;

        if (a_side != b_side )
        {
            GLvector dir = *b-*a;

            // planeIntersection
            float s = ((p0-*a)%n)/(dir % n);

            // TODO why [-.1, 1.1]?
            //if (!isnan(s) && -.1 <= s && s <= 1.1)
            if (!isnan(s) && 0 <= s && s <= 1)
            {
                break;
            }
        }
    }

    if (i==p.size())
    {
        if (!b_side)
            p.clear();

        return;
    }

    vector<GLvector> r;
    r.reserve(2*p.size());

    b = &p[p.size()-1];
    b_side = (p0-*b)%n < 0;

    for (unsigned i=0; i<p.size(); i++) {
        a = b;
        b = &p[i];

        a_side = b_side;
        b_side = (p0-*b)%n <0;

        if (a_side)
            r.push_back( *a );

        if (a_side != b_side )
        {
            float s;
            GLvector xy = planeIntersection( *a, *b, s, p0, n );

            //if (!isnan(s) && -.1 <= s && s <= 1.1)
            if (!isnan(s) && 0 <= s && s <= 1)
            {
                r.push_back( xy );
            }
        }
    }

    p = r;
}


vector<GLvector> FrustumClip::
        clipFrustum( vector<GLvector> l, GLvector &closest_i )
{
    //printl("Start",l);
    // Don't bother with projectionNormal?
    //clipPlane(l, projectionPlane, projectionNormal);
    //printl("Projectionclipped",l);
    clipPlane(l, rightPlane, rightNormal);
    //printl("Right", l);
    clipPlane(l, leftPlane, leftNormal);
    //printl("Left", l);
    clipPlane(l, topPlane, topNormal);
    //printl("Top",l);
    clipPlane(l, bottomPlane, bottomNormal);
    //printl("Bottom",l);
    //printl("Clipped polygon",l);

    closest_i = closestPointOnPoly(l, projectionPlane);
    return l;
}


vector<GLvector> FrustumClip::
        clipFrustum( GLvector corner[4], GLvector &closest_i )
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
