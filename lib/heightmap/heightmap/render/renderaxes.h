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
    struct Vertex {
        tvector<4,GLfloat> position;
        tvector<4,GLfloat> color;
    };

    struct Glyph {
        matrixd modelview;
        std::string text;
        double margin;
        double letter_spacing;
        double align_x;
        double align_y;
    };

    struct AxesElements {
        std::vector<tvector<4,GLfloat>> ticks;
        std::vector<tvector<4,GLfloat>> phatTicks;
        std::vector<Glyph> glyphs;
        std::vector<Vertex> vertices;
        std::vector<Vertex> orthovertices;
    };

    RenderAxes(
            const RenderSettings& render_settings,
            const glProjection* gl_projection,
            FreqAxis display_scale);

    void drawAxes( float T );

private:
    void getElements( RenderAxes::AxesElements& ae, float T );
    void drawElements( const AxesElements& );
    void drawGlyphsGlut( const std::vector<Glyph>& );

    const RenderSettings& render_settings;
    const glProjection* gl_projection;
    FreqAxis display_scale;
    AxesElements ae_;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERAXES_H
