#ifndef HEIGHTMAP_RENDER_RENDERREGION_H
#define HEIGHTMAP_RENDER_RENDERREGION_H

#include "heightmap/position.h"
#include "glprojection.h"

namespace Heightmap {
namespace Render {

class RenderRegion
{
public:
    RenderRegion(glProjection gl_projection);

    void render(Region r, bool drawcross=true);
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERREGION_H
