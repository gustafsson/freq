#ifndef HEIGHTMAP_RENDER_RENDERAXES_H
#define HEIGHTMAP_RENDER_RENDERAXES_H

#include "rendersettings.h"
#include "frustumclip.h"
#include "heightmap/freqaxis.h"

#include "glprojection.h"

#include <vector>

namespace Heightmap {
namespace Render {

class RenderAxes
{
public:
    RenderAxes(
            const RenderSettings& render_settings,
            const glProjection* gl_projection,
            FreqAxis display_scale);

    void drawAxes( float T );

private:
    const RenderSettings& render_settings;
    const glProjection* gl_projection;
    FreqAxis display_scale;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERAXES_H
