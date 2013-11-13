#ifndef HEIGHTMAP_RENDER_RENDERAXES_H
#define HEIGHTMAP_RENDER_RENDERAXES_H

#include "../rendersettings.h"
#include "frustumclip.h"
#include "tfr/freqaxis.h"

#include "glprojection.h"

#include <vector>

namespace Heightmap {
namespace Render {

class Axes
{
public:
    virtual std::vector<GLvector> getClippedFrustum()=0;
};

class RenderAxes: public Axes
{
public:
    RenderAxes(
            RenderSettings& render_settings,
            glProjection* gl_projection,
            Render::FrustumClip* frustum_clip,
            Tfr::FreqAxis display_scale);

    void drawAxes( float T );

    // Axes
    std::vector<GLvector> getClippedFrustum();

private:
//    void frustumMinMaxT( float& min_t, float& max_t);
    std::vector<GLvector> clippedFrustum;

    RenderSettings& render_settings;
    glProjection* gl_projection;
    Render::FrustumClip* frustum_clip;
    Tfr::FreqAxis display_scale;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERAXES_H
