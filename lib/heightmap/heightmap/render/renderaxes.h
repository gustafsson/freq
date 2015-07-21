#ifndef HEIGHTMAP_RENDER_RENDERAXES_H
#define HEIGHTMAP_RENDER_RENDERAXES_H

#include "rendersettings.h"
#include "frustumclip.h"
#include "heightmap/freqaxis.h"
#include "tvector.h"

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
    struct Glyph {
        matrixd modelview;
        std::string text;
        float margin;
        float letter_spacing;
        float align_x;
        float align_y;
    };

    struct AxesElements {
        std::vector<tvector<4,GLfloat>> ticks;
        std::vector<tvector<4,GLfloat>> phatTicks;
        std::vector<Glyph> glyphs;
    };

    AxesElements getGlyphs( float T );
    void drawGlyphsGlut( const AxesElements& );

    const RenderSettings& render_settings;
    const glProjection* gl_projection;
    FreqAxis display_scale;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERAXES_H
