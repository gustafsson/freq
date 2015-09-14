#include "geometricalgebra.h"
#include <cmath>
#include <float.h>

using namespace std;

namespace GeometricAlgebra {

vectord::baseT distanceToPlane( vectord obj, const vectord& plane, const vectord& normal ) {
    return (plane-obj)%normal;
}


vectord closestPointOnPoly( const vector<vectord>& l, const vectord &target)
{
    if (l.empty ())
        return target;

    vectord r;
    // check if point lies inside
    bool allNeg = true, allPos = true;

    // find point on poly boundary closest to 'target'
    double min = DBL_MAX;
    for (unsigned i2=0, i=l.size ()-1, i0=(2*l.size ()-2) % l.size();
         i2<l.size();
         i0=i, i=i2++)
    {
        vectord d = (l[i2] - l[i]),
                 v = target - l[i],
                 n = d ^ (l[i0] - l[i]);

        double f = v.dot ();
        if (f<min) {
            min = f;
            r = l[i];
        }

        if (0==d.dot())
            continue;

        double s = (d^v) % n;
        if (s < 0) allNeg=false;
        if (s > 0) allPos=false;
        double k = d%v / d.dot();
        if (0<k && k<1) {
            vectord t = l[i] + d*k;
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
            vectord n = (l[0]-l[1])^(l[0]-l[2]);
            if (0 != n.dot()) {
                n = n.Normalized();
                r = target + n*distanceToPlane( target, l[0], n );
            }
        }
    }
    return r;
}


vectord planeIntersection( vectord const& pt1, vectord const& pt2, double &s, const tvector<4,double>& plane) {
    vectord dir = pt2-pt1;
    vectord normal = {plane[0], plane[1], plane[2]};

    double denom = dir % normal;
    s = (-plane[3] - (pt1 % normal)) / denom;
    vectord p = pt1 + dir * s;

    return p;
}

vector<vectord> clipPlane( const vector<vectord>& p, const vectord& p0, const vectord& n )
{
    tvector<4,double> plane(n[0], n[1], n[2], -(p0 % n));
    return clipPlane( p, plane );
}

std::vector<vectord> clipPlane( const std::vector<vectord>& p, const tvector<4,double>& plane )
{
    if (p.empty())
        return vector<vectord>();

    vectord n(plane[0], plane[1], plane[2]);
    double d = -plane[3];

    unsigned i;

    vectord const* a, * b = &p[p.size()-1];
    bool a_side, b_side = d - (*b % n) < 0;
    for (i=0; i<p.size(); i++) {
        a = b;
        b = &p[i];

        a_side = b_side;
        b_side = d - (*b % n) < 0;

        if (a_side != b_side )
        {
            vectord dir = *b-*a;

            // planeIntersection
            double s = (d - (*a % n))/(dir % n);

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
            return vector<vectord>();
        else
            return p; // copy
    }

    vector<vectord> r;
    r.reserve(2*p.size());

    b = &p[p.size()-1];
    b_side = d - (*b % n) < 0;

    for (unsigned i=0; i<p.size(); i++) {
        a = b;
        b = &p[i];

        a_side = b_side;
        b_side = d - (*b % n) <0;

        if (a_side)
            r.push_back( *a );

        if (a_side != b_side )
        {
            double s;
            vectord xy = planeIntersection( *a, *b, s, plane );

            if (!isnan(s) && -.1 <= s && s <= 1.1)
            //if (!isnan(s) && 0 <= s && s <= 1)
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
        vectord obj, plane, normal;
        double d;
        obj = vectord(0,0,0);
        plane = vectord(1,1,0);
        normal = vectord(1,1,0);
        d = distanceToPlane( obj, plane, normal );

        EXCEPTION_ASSERT_EQUALS(d, 2);

        d = distanceToPlane( vectord(1,1,1), vectord(2,2,2), vectord(1,-1,1) );
        EXCEPTION_ASSERT_EQUALS(d, 1);

        d = distanceToPlane( vectord(1,1,1), vectord(2,2,2), vectord(1,1,1) );
        EXCEPTION_ASSERT_EQUALS(d, 3);

        d = distanceToPlane( vectord(1,1,1), vectord(2,2,2), vectord(1,1,1).Normalized() );
        EXCEPTION_ASSERT_EQUALS(float(d), sqrtf(3.f));
    }

    {
        // closestPointOnPoly computes the closest point 'r' in a polygon.
        std::vector<vectord> l = {
            vectord(0,0,0),
            vectord(0,1,0),
            vectord(1,0,0)};
        vectord c = closestPointOnPoly( l, vectord(2, 2, 0));
        EXCEPTION_ASSERT_EQUALS(c, vectord(0.5, 0.5, 0));

        c = closestPointOnPoly( l, vectord(0.9, 0.9, 0));
        EXCEPTION_ASSERT_EQUALS(c, vectord(0.5, 0.5, 0));

        c = closestPointOnPoly( l, vectord(1, 1, 0));
        EXCEPTION_ASSERT_EQUALS(c, vectord(0.5, 0.5, 0));

        c = closestPointOnPoly( l, vectord(1.1, 1.1, 0));
        EXCEPTION_ASSERT_EQUALS(c, vectord(0.5, 0.5, 0));

        l = std::vector<vectord>{vectord(1,0,0)};
        c = closestPointOnPoly( l, vectord(1.1, 1.1, 0));
        EXCEPTION_ASSERT_EQUALS(c, vectord(1, 0, 0));

        l = std::vector<vectord>{vectord(1,0,0), vectord(1,0,0)};
        c = closestPointOnPoly( l, vectord(1.1, 1.1, 0));
        EXCEPTION_ASSERT_EQUALS(c, vectord(1, 0, 0));

        l = std::vector<vectord>{
            vectord(0,0,0),
            vectord(0,1,0),
            vectord(0,1,0),
            vectord(0,1,0),
            vectord(0,1,0),
            vectord(1,0,0)};
        c = closestPointOnPoly( l, vectord(1.1, 1.1, 0));
        EXCEPTION_ASSERT_EQUALS(c, vectord(0.5, 0.5, 0));
    }

    {
        // planeIntersection computes the intersection of a line and a plane
        vectord pt1(1,0,0);
        vectord pt2(3,0,0);
        double s = 0.f/0.f;
        vectord plane(2,1,1);
        vectord normal(1,0,0);
        tvector<4,double> comb(1, 0, 0, -plane % normal);
        vectord p = planeIntersection( pt1, pt2, s, comb );
        EXCEPTION_ASSERT_EQUALS(p, vectord(2, 0, 0));
        EXCEPTION_ASSERT_EQUALS(s, 0.5);
    }

    {
        // clipPlane clips a polygon with a plane.
        std::vector<vectord> r, p = {
            vectord(0,0,0),
            vectord(0,1,0),
            vectord(1,0,0)};
        vectord p0(0.5,0,0);
        vectord n(1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 3u);
        EXCEPTION_ASSERT_EQUALS(r[0], vectord(1,0,0));
        EXCEPTION_ASSERT_EQUALS(r[1], vectord(0.5,0,0));
        EXCEPTION_ASSERT_EQUALS(r[2], vectord(0.5,0.5,0));

        p0 = vectord(0,0,0);
        n = vectord(1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 3u);
        EXCEPTION_ASSERT_EQUALS(r[0], vectord(1,0,0));
        EXCEPTION_ASSERT_EQUALS(r[1], vectord(0,0,0));
        EXCEPTION_ASSERT_EQUALS(r[2], vectord(0,1,0));

        p0 = vectord(1,0,0);
        n = vectord(1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 0u);

        p0 = vectord(0,0,0);
        n = vectord(-1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 0u);

        p0 = vectord(0.5,0,0);
        n = vectord(-1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 4u);
        EXCEPTION_ASSERT_EQUALS(r[0], vectord(0.5,0,0));
        EXCEPTION_ASSERT_EQUALS(r[1], vectord(0,0,0));
        EXCEPTION_ASSERT_EQUALS(r[2], vectord(0,1,0));
        EXCEPTION_ASSERT_EQUALS(r[3], vectord(0.5,0.5,0));

        p0 = vectord(2,0,0);
        n = vectord(1,0,0);
        r = clipPlane( p, p0, n );
        EXCEPTION_ASSERT_EQUALS(r.size (), 0u);
    }
}

} // namespace GeometricAlgebra
