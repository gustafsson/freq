#ifndef HEIGHTMAP_RENDER_RENDERFRUSTUM_H
#define HEIGHTMAP_RENDER_RENDERFRUSTUM_H

#include "glprojection.h"

namespace Heightmap {
namespace Render {

class RenderFrustum
{
public:
    RenderFrustum(const glProjection& gl_projection);

    // More like, drawExtentOfPrimaryViewportInSecondaryViewport
    void drawFrustum();

private:
    vectord camera;
    std::vector<vectord> clippedFrustum;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERFRUSTUM_H
