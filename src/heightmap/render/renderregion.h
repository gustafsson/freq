#ifndef HEIGHTMAP_RENDER_RENDERREGION_H
#define HEIGHTMAP_RENDER_RENDERREGION_H

#include "heightmap/position.h"

namespace Heightmap {
namespace Render {

class RenderRegion
{
public:
    RenderRegion(Region r);

    void render();

private:
    Region r;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERREGION_H
