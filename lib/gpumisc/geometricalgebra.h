#ifndef GEOMETRICALGEBRA_H
#define GEOMETRICALGEBRA_H

#include "GLvector.h"
#include <vector>


/**
 * The GeometricAlgebra namespace should compute things with planes and vectors.
 */
namespace GeometricAlgebra
{

/**
 * @brief distanceToPlane computes the distance to a point from a plane.
 * @param obj A point.
 * @param plane, normal Defines the plane.
 * @return The distance measured in units along 'normal', so a negative
 * distance means obj is in front of the plane.
 */
vectord::baseT distanceToPlane( vectord obj, const vectord& plane, const vectord& normal );


/**
 * @brief closestPointOnPoly computes the closest point 'r' in a polygon.
 * @param l Defines the polygon. Must be convex and non-empty.
 * All points in 'l' are assumed to be in the same plane.
 * @param target Minimize '|r-target|', where 'r' is a point in the polygon.
 * @return 'r'. If the polygon 'l' is empty, returns target.
 */
vectord closestPointOnPoly( const std::vector<vectord>& l, const vectord &target);


/**
 * @brief planeIntersection computes the intersection of a line and a plane.
 * @param pt1, pt2 Defines the line.
 * @param s [out] The distance along the line to the intersection. Such that
 * planeIntersection(...) = pt1 + (pt2-pt1) * s.
 * @param plane, normal Defines the plane.
 * The normal does not need to be normalized.
 * @return A point in the plane. Perpendicular line and normal leads to division by 0.
 */
vectord planeIntersection( vectord const& pt1, vectord const& pt2, double &s, const tvector<4,double>& plane );


/**
 * @brief clipPlane clips a polygon with a plane.
 * @param p Defines the polygon.
 * @param p0,n Defines the plane.
 * @return The polygon on the frontside of the plane.
 */
void clipPlane( std::vector<vectord>& p, const vectord& p0, const vectord& n );
void clipPlane( std::vector<vectord>& p, const tvector<4,double>& plane );

void test();

}

#endif // GEOMETRICALGEBRA_H
