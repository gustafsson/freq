#include "geometricalgebra.h"
#include <cmath>
#include <float.h>

using namespace std;

namespace GeometricAlgebra {

GLvector::baseT distanceToPlane( GLvector obj, const GLvector& plane, const GLvector& normal ) {
    return (plane-obj)%normal;
}


GLvector closestPointOnPoly( const vector<GLvector>& l, const GLvector &target)
{
    if (l.empty ())
        return target;

    GLvector r;
    // check if point lies inside
    bool allNeg = true, allPos = true;

    // find point on poly boundary closest to 'target'
    float min = FLT_MAX;
    for (unsigned i2=0, i=l.size ()-1, i0=(2*l.size ()-2) % l.size();
         i2<l.size();
         i0=i, i=i2++)
    {
        GLvector d = (l[i2] - l[i]),
                 v = target - l[i],
                 n = d ^ (l[i0] - l[i]);

        float f = v.dot ();
        if (f<min) {
            min = f;
            r = l[i];
        }

        if (0==d.dot())
            continue;

        float s = (d^v) % n;
        if (s < 0) allNeg=false;
        if (s > 0) allPos=false;
        float k = d%v / d.dot();
        if (0<k && k<1) {
            GLvector t = l[i] + d*k;
            f = (target - t).dot();
            if (f<min) {
                min = f;
                r = t;
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


GLvector planeIntersection( GLvector const& pt1, GLvector const& pt2, float &s, GLvector const& plane, GLvector const& normal ) {
    GLvector dir = pt2-pt1;

    s = ((plane-pt1)%normal)/(dir % normal);
    GLvector p = pt1 + dir * s;

    return p;
}


vector<GLvector> clipPlane( const vector<GLvector>& p, const GLvector& p0, const GLvector& n )
{
    if (p.empty())
        return vector<GLvector>();

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
            return vector<GLvector>();
        else
            return p; // copy
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

    return r;
}

} // namespace GeometricAlgebra

#include "exceptionassert.h"
#include "tvectorstring.h"

namespace GeometricAlgebra {

void test() {
    // The GeometricAlgebra namespace should compute things with planes and vectors.
    {
        // distanceToPlane computes the distance to a point from a plane.
        GLvector obj, plane, normal;
        float d;
        obj = GLvector(0,0,0);
        plane = GLvector(1,1,0);
        normal = GLvector(1,1,0);
        d = distanceToPlane( obj, plane, normal );

        EXCEPTION_ASSERT_EQUALS(d, 2);

        d = distanceToPlane( GLvector(1,1,1), GLvector(2,2,2), GLvector(1,-1,1) );
        EXCEPTION_ASSERT_EQUALS(d, 1);

        d = distanceToPlane( GLvector(1,1,1), GLvector(2,2,2), GLvector(1,1,1) );
        EXCEPTION_ASSERT_EQUALS(d, 3);

        d = distanceToPlane( GLvector(1,1,1), GLvector(2,2,2), GLvector(1,1,1).Normalized() );
        EXCEPTION_ASSERT_EQUALS(d, sqrtf(3.f));
    }

    {
        // closestPointOnPoly computes the closest point 'r' in a polygon.
        std::vector<GLvector> l = {
            GLvector(0,0,0),
            GLvector(0,1,0),
            GLvector(1,0,0)};
        GLvector c = closestPointOnPoly( l, GLvector(2, 2, 0));
        EXCEPTION_ASSERT_EQUALS(c, GLvector(0.5, 0.5, 0));

        c = closestPointOnPoly( l, GLvector(0.9, 0.9, 0));
        EXCEPTION_ASSERT_EQUALS(c, GLvector(0.5, 0.5, 0));

        c = closestPointOnPoly( l, GLvector(1, 1, 0));
        EXCEPTION_ASSERT_EQUALS(c, GLvector(0.5, 0.5, 0));

        c = closestPointOnPoly( l, GLvector(1.1, 1.1, 0));
        EXCEPTION_ASSERT_EQUALS(c, GLvector(0.5, 0.5, 0));

        l = std::vector<GLvector>{GLvector(1,0,0)};
        c = closestPointOnPoly( l, GLvector(1.1, 1.1, 0));
        EXCEPTION_ASSERT_EQUALS(c, GLvector(1, 0, 0));

        l = std::vector<GLvector>{GLvector(1,0,0), GLvector(1,0,0)};
        c = closestPointOnPoly( l, GLvector(1.1, 1.1, 0));
        EXCEPTION_ASSERT_EQUALS(c, GLvector(1, 0, 0));

        l = std::vector<GLvector>{
            GLvector(0,0,0),
            GLvector(0,1,0),
            GLvector(0,1,0),
            GLvector(0,1,0),
            GLvector(0,1,0),
            GLvector(1,0,0)};
        c = closestPointOnPoly( l, GLvector(1.1, 1.1, 0));
        EXCEPTION_ASSERT_EQUALS(c, GLvector(0.5, 0.5, 0));
    }

    {
        // planeIntersection computes the intersection of a line and a plane
        GLvector pt1(1,0,0);
        GLvector pt2(3,0,0);
        float s = 0.f/0.f;
        GLvector plane(2,1,1);
        GLvector normal(1,0,0);
        GLvector p = planeIntersection( pt1, pt2, s, plane, normal );
        EXCEPTION_ASSERT_EQUALS(p, GLvector(2, 0, 0));
        EXCEPTION_ASSERT_EQUALS(s, 0.5);
    }

    {
        // clipPlane clips a polygon with a plane.
        std::vector<GLvector> r, p = {
            GLvector(0,0,0),
            GLvector(0,1,0),
            GLvector(1,0,0)};
        GLvector p0(0.5,0,0);
        GLvector n(1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 3);
        EXCEPTION_ASSERT_EQUALS(r[0], GLvector(1,0,0));
        EXCEPTION_ASSERT_EQUALS(r[1], GLvector(0.5,0,0));
        EXCEPTION_ASSERT_EQUALS(r[2], GLvector(0.5,0.5,0));

        p0 = GLvector(0,0,0);
        n = GLvector(1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 3);
        EXCEPTION_ASSERT_EQUALS(r[0], GLvector(1,0,0));
        EXCEPTION_ASSERT_EQUALS(r[1], GLvector(0,0,0));
        EXCEPTION_ASSERT_EQUALS(r[2], GLvector(0,1,0));

        p0 = GLvector(1,0,0);
        n = GLvector(1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 0);

        p0 = GLvector(0,0,0);
        n = GLvector(-1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 0);

        p0 = GLvector(0.5,0,0);
        n = GLvector(-1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 4);
        EXCEPTION_ASSERT_EQUALS(r[0], GLvector(0.5,0,0));
        EXCEPTION_ASSERT_EQUALS(r[1], GLvector(0,0,0));
        EXCEPTION_ASSERT_EQUALS(r[2], GLvector(0,1,0));
        EXCEPTION_ASSERT_EQUALS(r[3], GLvector(0.5,0.5,0));

        p0 = GLvector(2,0,0);
        n = GLvector(1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 0);
    }
}

} // namespace GeometricAlgebra
