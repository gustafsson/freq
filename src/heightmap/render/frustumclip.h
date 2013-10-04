#ifndef HEIGHTMAP_RENDER_FRUSTUMCLIP_H
#define HEIGHTMAP_RENDER_FRUSTUMCLIP_H

#include "glprojection.h"
#include <vector>

namespace Heightmap {
namespace Render {

class FrustumClip
{
public:
    FrustumClip(glProjection* gl_projection, bool* left_handed_axes);

    /**
     * @brief update
     * @param w == ?
     * @param h == ?
     */
    void update(float w, float h);
    const std::vector<GLvector> clippedFrustum() const;

    std::vector<GLvector> clipFrustum( GLvector corner[4], GLvector &closest_i );
    std::vector<GLvector> clipFrustum( std::vector<GLvector> l, GLvector &closest_i );

    GLvector projectionPlane, projectionNormal; // for clipFrustum

private:
    GLvector rightPlane, rightNormal,
        leftPlane, leftNormal,
        topPlane, topNormal,
        bottomPlane, bottomNormal;

    glProjection* gl_projection;
    bool* left_handed_axes;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_FRUSTUMCLIP_H
