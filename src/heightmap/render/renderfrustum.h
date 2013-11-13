#ifndef HEIGHTMAP_RENDER_RENDERFRUSTUM_H
#define HEIGHTMAP_RENDER_RENDERFRUSTUM_H

#include "renderaxes.h"
#include "../rendersettings.h"

namespace Heightmap {
namespace Render {

class RenderFrustum
{
public:
    RenderFrustum(RenderSettings& render_settings, std::vector<GLvector> clippedFrustum);

    // More like, drawExtentOfPrimaryViewportInSecondaryViewport
    void drawFrustum();

private:
    RenderSettings& render_settings;
    std::vector<GLvector> clippedFrustum;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERFRUSTUM_H
