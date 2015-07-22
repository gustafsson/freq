#ifndef HEIGHTMAP_RENDER_RENDERAXES_H
#define HEIGHTMAP_RENDER_RENDERAXES_H

#include "rendersettings.h"
#include "frustumclip.h"
#include "heightmap/freqaxis.h"
#include "tvector.h"
#include "glyphs.h"

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

    struct AxesElements {
        std::vector<GlyphData> glyphs;
        std::vector<Vertex> vertices;
        std::vector<Vertex> orthovertices;
    };

    RenderAxes(
            const RenderSettings& render_settings,
            const glProjection* gl_projection,
            FreqAxis display_scale);
    ~RenderAxes();

    void drawAxes( float T );

private:
    void getElements( RenderAxes::AxesElements& ae, float T );
    void drawElements( const AxesElements& );

    const RenderSettings& render_settings;
    const glProjection* gl_projection;
    FreqAxis display_scale;
    AxesElements ae_;
    QOpenGLShaderProgram* program_;
    IGlyphs* glyphs_;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERAXES_H
