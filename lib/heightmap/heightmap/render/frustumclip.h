#ifndef HEIGHTMAP_RENDER_FRUSTUMCLIP_H
#define HEIGHTMAP_RENDER_FRUSTUMCLIP_H

#include "glprojection.h"
#include <vector>

namespace Heightmap {
namespace Render {

class FrustumClip
{
public:
    FrustumClip(const glProjection& gl_projection, float border_width=0, float border_height=0);

    const GLvector& getCamera() const { return camera; }

    std::vector<GLvector> clipFrustum( GLvector corner[4], GLvector* closest_i=0 ) const;
    std::vector<GLvector> clipFrustum( std::vector<GLvector> l, GLvector* closest_i=0 ) const;
    std::vector<GLvector> visibleXZ();

private:
    GLvector camera;
    tvector<4,GLfloat> right, left, top, bottom, far, near;

    void update(const glProjection& gl_projection, float w, float h);
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_FRUSTUMCLIP_H
